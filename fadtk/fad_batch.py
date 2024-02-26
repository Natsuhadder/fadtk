import multiprocessing
import os
from pathlib import Path
from typing import Callable, Union
import numpy as np
import shutil
import torch

from .fad import log, FrechetAudioDistance
from .model_loader import ModelLoader
from .utils import get_cache_embedding_path

def _cache_embedding_batch(args):
    fs: list[Path]
    ml: ModelLoader
    fs, ml, kwargs = args
    fad = FrechetAudioDistance(ml, **kwargs)
    for f in fs:
        log.info(f"Loading {f} using {ml.name}")
        fad.cache_embedding_file(f)


def cache_embedding_files(files: Union[list[Path], str, Path], ml: ModelLoader, workers: int = 8, force_emb_calc=False, **kwargs):
    """
    Get embeddings for all audio files in a directory.

    :param ml_fn: A function that returns a ModelLoader instance.
    """
    if isinstance(files, (str, Path)):
        files = list(Path(files).glob('*.*'))

    if force_emb_calc:
        emb_path = files[0].parent / "embeddings" / ml.name
        if os.path.exists(emb_path):
            # Remove the folder and its contents
            shutil.rmtree(emb_path)
            print(f"The folder '{emb_path}' has been successfully removed.")
        

    # Filter out files that already have embeddings
    files = [f for f in files if not get_cache_embedding_path(ml.name, f).exists()]

    if len(files) == 0:
        log.info("All files already have embeddings, skipping.")
        return

    log.info(f"[Frechet Audio Distance] Loading {len(files)} audio files...")

    # Split files into batches
    batches = list(np.array_split(files, workers))
    
    # Cache embeddings in parallel
    multiprocessing.set_start_method('spawn', force=True)
    with torch.multiprocessing.Pool(workers) as pool:
        pool.map(_cache_embedding_batch, [(b, ml, kwargs) for b in batches])
