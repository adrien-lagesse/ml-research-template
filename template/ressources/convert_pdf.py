"""Convert a catalogued paper's PDF into agent-readable markdown with docling.

Slug-driven: the script reads ``ressources.json`` next to it, looks up an entry
by its ``slug``, obtains the PDF (downloaded from the entry's ``url`` or supplied
by hand with ``--pdf``), and writes ``_input.pdf``, ``_output.md`` and an
``images/`` folder under ``ressources/<slug>/``. The ``summary.md`` step is left
to the reading agent; see ``README.md``.

Run it through uv, never bare python::

    uv run ressources/convert_pdf.py simclr
    uv run ressources/convert_pdf.py --all
    uv run ressources/convert_pdf.py dino --pdf ~/Downloads/dino.pdf
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import urllib.request

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption
from docling_core.types.doc import ImageRefMode

ROOT = Path(__file__).parent
CATALOG = ROOT / "ressources.json"
INPUT_PDF = "_input.pdf"
OUTPUT_MD = "_output.md"
IMAGES_DIR = "images"
IMAGES_SCALE = 2.0


@dataclass(frozen=True)
class Resource:
    """One catalogued paper.

    Attributes:
        slug: Unique kebab-case key; also the folder name under ``ressources/``.
        title: Paper title.
        url: Direct link to the PDF.
        date: Publication date, ISO ``YYYY-MM-DD``.
        repo: Code repository, or ``None`` when the paper has none.
    """

    slug: str
    title: str
    url: str
    date: str
    repo: str | None


def load_catalog(path: Path) -> list[Resource]:
    """Read the catalog file into a list of resources.

    Args:
        path: Path to ``ressources.json``.

    Returns:
        Every entry as a :class:`Resource`, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"catalog not found: {path}")
    entries = json.loads(path.read_text())
    return [
        Resource(
            slug=entry["slug"],
            title=entry["title"],
            url=entry["url"],
            date=entry["date"],
            repo=entry.get("repo"),
        )
        for entry in entries
    ]


def find_resource(catalog: list[Resource], slug: str) -> Resource:
    """Return the resource with the given slug.

    Args:
        catalog: Every catalogued resource.
        slug: The slug to look up.

    Returns:
        The matching resource.

    Raises:
        KeyError: If no entry has that slug.
    """
    for resource in catalog:
        if resource.slug == slug:
            return resource
    known = ", ".join(r.slug for r in catalog)
    raise KeyError(f"unknown slug {slug!r}; catalog has: {known}")


def download_pdf(url: str, dest: Path) -> None:
    """Download a PDF to ``dest``.

    Args:
        url: Direct link to the PDF.
        dest: Destination file (typically ``<slug>/_input.pdf``).
    """
    # A default user agent so hosts that reject urllib's still serve the file.
    request = urllib.request.Request(
        url, headers={"User-Agent": "filao-ssl/convert_pdf"}
    )
    with urllib.request.urlopen(request) as response:
        dest.write_bytes(response.read())


def convert_pdf(pdf_path: Path, out_md: Path, images_dir: Path) -> None:
    """Convert a PDF to markdown with docling, extracting figures.

    Writes ``out_md`` and populates ``images_dir`` with the figures it
    references. Existing contents of ``images_dir`` are cleared first so a
    re-run doesn't leave stale files behind.

    Args:
        pdf_path: The source PDF.
        out_md: Where to write the markdown export.
        images_dir: Folder for the extracted figures, referenced from ``out_md``.
    """
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    options = PdfPipelineOptions()
    options.images_scale = IMAGES_SCALE
    options.generate_picture_images = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    result = converter.convert(pdf_path)
    result.document.save_as_markdown(
        out_md, artifacts_dir=images_dir, image_mode=ImageRefMode.REFERENCED
    )


def process(
    resource: Resource, root: Path, *, force: bool, local_pdf: Path | None
) -> None:
    """Build one resource folder: obtain the PDF, then convert it.

    Args:
        resource: The catalogued paper to process.
        root: The ``ressources/`` directory holding the slug folders.
        force: Re-download and re-convert even when outputs already exist.
        local_pdf: A hand-supplied PDF to use instead of downloading ``url``;
            for papers that block automatic downloads.

    Raises:
        FileNotFoundError: If ``local_pdf`` is given but does not exist.
    """
    folder = root / resource.slug
    folder.mkdir(parents=True, exist_ok=True)
    input_pdf = folder / INPUT_PDF

    if local_pdf is not None:
        if not local_pdf.exists():
            raise FileNotFoundError(f"local pdf not found: {local_pdf}")
        shutil.copyfile(local_pdf, input_pdf)
    elif force or not input_pdf.exists():
        print(f"[{resource.slug}] downloading {resource.url}")
        download_pdf(resource.url, input_pdf)

    output_md = folder / OUTPUT_MD
    if not force and output_md.exists():
        print(f"[{resource.slug}] {OUTPUT_MD} exists, skipping (use --force to redo)")
        return

    print(f"[{resource.slug}] converting {INPUT_PDF} -> {OUTPUT_MD}")
    convert_pdf(input_pdf, output_md, folder / IMAGES_DIR)
    print(f"[{resource.slug}] done; now write {folder / 'summary.md'}")


def main() -> None:
    """Parse arguments and process one slug or the whole catalog."""
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("slug", nargs="?", help="the catalog slug to convert")
    target.add_argument(
        "--all", action="store_true", help="convert every entry in the catalog"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="use this local PDF instead of downloading (single slug only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download and re-convert even if outputs exist",
    )
    args = parser.parse_args()

    if args.all and args.pdf is not None:
        parser.error("--pdf cannot be combined with --all")

    catalog = load_catalog(CATALOG)
    resources = catalog if args.all else [find_resource(catalog, args.slug)]
    for resource in resources:
        process(resource, ROOT, force=args.force, local_pdf=args.pdf)


if __name__ == "__main__":
    main()
