from collections import Counter
from pathlib import Path
import hashlib
import json

import numpy as np

from src.pdf_loader import load_pdf
from src.chunker import chunk_documents
from src.embedder import Embedder
from src.vector_store.store import VectorStore


class DocumentIndexer:

  def __init__(
    self,
    documents_dir: str = "data/documents",
    storage_dir: str = "data/embeddings",
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
  ):
    self.documents_dir = Path(documents_dir)
    self.storage_dir = Path(storage_dir)
    self.manifest_path = self.storage_dir / "manifest.json"

    self.embedder = embedder or Embedder()
    self.store = store or VectorStore()

  @property
  def _dimension(self) -> int:
    return self.embedder.model.get_embedding_dimension()

  # -----------------------------------------
  # Files
  # -----------------------------------------

  def _get_file_hash(self, file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
      for block in iter(lambda: file.read(65536), b""):
        sha256.update(block)

    return sha256.hexdigest()

  def _get_pdf_files(self) -> dict[str, Path]:
    return {
      path.name: path
      for path in sorted(self.documents_dir.iterdir())
      if path.is_file() and path.suffix.lower() == ".pdf"
    }

  # -----------------------------------------
  # Manifest
  # -----------------------------------------

  def _load_manifest(self) -> dict:
    if not self.manifest_path.exists():
      return {}

    with self.manifest_path.open("r", encoding="utf-8") as file:
      return json.load(file)

  def _save_manifest(self, manifest: dict) -> None:
    self.storage_dir.mkdir(parents=True, exist_ok=True)

    # Write to a temp file then swap, so a crash never leaves a half-written manifest.
    temp_path = self.manifest_path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
      json.dump(manifest, file, indent=4)

    temp_path.replace(self.manifest_path)

  # -----------------------------------------
  # Vector store
  # -----------------------------------------

  def _load_store(self) -> tuple[list[dict], np.ndarray, bool]:
    try:
      chunks, embeddings = self.store.load()
      return chunks, embeddings, True
    except FileNotFoundError:
      return [], np.empty((0, 0), dtype=np.float32), False

  # -----------------------------------------
  # Change detection
  # -----------------------------------------

  def _classify_files(self, current_files: dict, old_manifest: dict):
    """Hash every PDF and compare with the old manifest.

    Returns (new_manifest, new_files, modified_files). Unchanged files
    keep their previously recorded chunk_count.
    """
    manifest, new_files, modified_files = {}, [], []

    for name, path in current_files.items():
      file_hash = self._get_file_hash(path)
      old_entry = old_manifest.get(name)

      manifest[name] = {"sha256": file_hash, "chunk_count": 0}

      if old_entry and old_entry.get("sha256") == file_hash:
        manifest[name]["chunk_count"] = old_entry.get("chunk_count", 0)
      elif old_entry:
        modified_files.append(name)
      else:
        new_files.append(name)

    return manifest, new_files, modified_files

  def _find_deleted(self, old_manifest: dict, chunks: list[dict], current_files: dict) -> list[str]:
    """Files known to the manifest or the store that are no longer on disk."""
    stored_sources = {chunk["source"] for chunk in chunks}
    return sorted((set(old_manifest) | stored_sources) - set(current_files))

  def _find_incomplete(self, manifest: dict, chunks: list[dict], skip: set[str]) -> list[str]:
    """Unchanged files whose stored chunk count differs from the manifest."""
    stored_counts = Counter(chunk["source"] for chunk in chunks)

    return [
      name
      for name, entry in manifest.items()
      if name not in skip and entry["chunk_count"] != stored_counts[name]
    ]

  # -----------------------------------------
  # Processing
  # -----------------------------------------

  def _process_document(self, pdf_path: Path, file_hash: str):
    chunks = chunk_documents(load_pdf(str(pdf_path)))

    for chunk in chunks:
      chunk["document_hash"] = file_hash

    if not chunks:
      return chunks, np.empty((0, self._dimension), dtype=np.float32)

    return chunks, self.embedder.generate_embedding([c["text"] for c in chunks])

  def _keep_unchanged(self, chunks: list[dict], embeddings: np.ndarray, drop: set[str]):
    """Reuse stored chunks and embeddings for every file not in `drop`."""
    keep = [i for i, chunk in enumerate(chunks) if chunk["source"] not in drop]

    kept_chunks = [chunks[i] for i in keep]
    parts = [embeddings[keep]] if keep else []

    return kept_chunks, parts

  def _index_files(self, names: list[str], current_files: dict, manifest: dict):
    """Chunk and embed each file; records chunk_count in `manifest`."""
    all_chunks, parts = [], []

    for name in names:
      print(f"\nIndexing: {name}")

      chunks, embeddings = self._process_document(
        current_files[name], manifest[name]["sha256"]
      )

      manifest[name]["chunk_count"] = len(chunks)
      all_chunks.extend(chunks)

      if len(embeddings):
        parts.append(embeddings)

    return all_chunks, parts

  # -----------------------------------------
  # Synchronisation
  # -----------------------------------------

  def sync(self) -> dict:
    self.documents_dir.mkdir(parents=True, exist_ok=True)
    self.storage_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load previous state.
    chunks, embeddings, store_loaded = self._load_store()
    manifest_exists = self.manifest_path.exists()
    old_manifest = self._load_manifest()

    # 2. Work out what changed.
    current_files = self._get_pdf_files()
    manifest, new_files, modified_files = self._classify_files(current_files, old_manifest)

    # Store and manifest must agree; if either is missing, rebuild everything.
    if not store_loaded or not manifest_exists:
      new_files, modified_files = list(current_files), []

    changed = set(new_files) | set(modified_files)
    deleted_files = self._find_deleted(old_manifest, chunks, current_files)

    # Integrity: unchanged files whose chunks went missing get re-indexed.
    incomplete = self._find_incomplete(manifest, chunks, skip=changed)
    modified_files += incomplete
    changed.update(incomplete)

    # 3. Keep valid data, re-index the rest.
    updated_chunks, parts = self._keep_unchanged(
      chunks, embeddings, drop=changed | set(deleted_files)
    )
    new_chunks, new_parts = self._index_files(sorted(changed), current_files, manifest)

    updated_chunks += new_chunks
    parts += new_parts

    updated_embeddings = (
      np.concatenate(parts, axis=0)
      if parts
      else np.empty((0, self._dimension), dtype=np.float32)
    )

    # 4. Persist: vectors first, manifest last.
    self.store.save(updated_chunks, updated_embeddings)
    self._save_manifest(manifest)

    return {
      "added": new_files,
      "modified": modified_files,
      "deleted": deleted_files,
      "total_documents": len(manifest),
      "total_chunks": len(updated_chunks),
    }