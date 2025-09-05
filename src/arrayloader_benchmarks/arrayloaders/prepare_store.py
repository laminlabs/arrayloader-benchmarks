from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import lamindb as ln
import zarr
import zarrs  # noqa
from arrayloaders.io import create_store_from_h5ads
from zarr.codecs import BloscCodec, BloscShuffle

if TYPE_CHECKING:
    from typing import Literal


zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline", "threading.max_workers": None}
)

ln.settings.sync_git_repo = "https://github.com/laminlabs/arrayloader-benchmarks"
ln.track(project="zjQ6EYzMXif4")

# Set path where to save the output store below
STORE_PATH: Path = Path("/dss/mcmlscratch/04/di93zer/tahoe100_FULL")

GENE_SPACE: Literal["FULL", "PROTEIN_CODING"] = "FULL"
N_SHARDS_INPUT: int = 48
assert 0 < N_SHARDS_INPUT <= 48, "N_SHARDS_INPUT must be between 1 and 48."
SHOULD_DENSIFY: bool = False
ZARR_CHUNK_SIZE: int = 32768
ZARR_SHARD_SIZE: int = 134_217_728
ANNDATA_SHARD_SIZE: int = 2_097_152
COMPRESSOR = BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle)
UPLOAD_TO_LAMINDB = True


if __name__ == "__main__":
    if GENE_SPACE == "FULL":
        h5ads_paths = ln.Collection.get("eAgoduHMxuDs5Wem0000").cache()[0].parent
    elif GENE_SPACE == "PROTEIN_CODING":
        h5ads_paths = ln.Collection.get("k9GqakN96EmLjn1L0000").cache()[0].parent
    else:
        err_msg = (
            f"Invalid gene space: {GENE_SPACE}. Must be 'FULL' or 'PROTEIN_CODING'."
        )
        raise ValueError(err_msg)

    create_store_from_h5ads(
        [h5ads_paths / f"shard_{i}.h5ad" for i in range(N_SHARDS_INPUT)],
        STORE_PATH,
        chunk_size=ZARR_CHUNK_SIZE,
        shard_size=ZARR_SHARD_SIZE,
        buffer_size=ANNDATA_SHARD_SIZE,
        zarr_compressor=(COMPRESSOR,),
        shuffle=False,
        should_denseify=SHOULD_DENSIFY,
    )

    if UPLOAD_TO_LAMINDB:
        artifacts = [
            ln.Artifact.from_anndata(shard, key=shard.name).save()
            for shard in STORE_PATH.iterdir()
            if shard.name.endswith(".zarr")
        ]
        ln.Collection(
            artifacts,
            key="Tahoe100M",
            description="Tahoe100M for arrayloader-benchmarks",
        )
        ln.Collection(
            [artifacts[0]],
            key="Tahoe100M_mini",
            description="Tahoe100M for arrayloader-benchmarks subset to 2mio cells",
        )

ln.finish()
