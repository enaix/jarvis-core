import enum
import gzip
import json
import os

import numpy as np
from PIL import Image


class FileType(enum.StrEnum):
    """
    Enum containing all filetypes
    """

    AXTree = "axtree.json.gz"
    BB = "bb.json.gz"
    Box = "box.json.gz"
    Class = "class.json.gz"
    HTML = "html.html"
    Links = "links.json"
    ScreenFull = "screenshot-full.webp"
    Screen = "screenshot.webp"
    Style = "style.json.gz"
    URL = "url.txt"
    Viewport = "viewport.json.gz"


def ft_is_gz(ft: FileType) -> bool:
    """
    Check if a FileType is a gz file
    """
    return ft in [
        FileType.AXTree,
        FileType.BB,
        FileType.Box,
        FileType.Class,
        FileType.Style,
        FileType.Viewport,
    ]


def ft_is_json(ft: FileType) -> bool:
    """
    Check if a FileType is in json format
    """
    return ft in [
        FileType.AXTree,
        FileType.BB,
        FileType.Box,
        FileType.Class,
        FileType.Links,
        FileType.Style,
        FileType.Viewport,
    ]


def ft_is_webp(ft: FileType) -> bool:
    """
    Check if a FileType is a webp image
    """
    return ft in [FileType.ScreenFull, FileType.Screen]


class Page:
    """
    A class which loads page content (screen)
    """

    def __init__(self, path, screen_type: str, debug: bool = True):
        self.screen_type = screen_type
        self.path = path
        self.skip = False

        # Parse screen type
        self.desktop = screen_type.startswith("default")
        if self.desktop:
            (self.width, self.height) = tuple(
                map(int, screen_type.split("_")[1].split("-"))
            )
        else:
            (self.width, self.height) = (0, 0)

        # Load filenames
        self.fnames = dict()
        self.files = dict()
        missing = False

        for ft in FileType:
            file_path = screen_type + "-" + ft.value
            if os.path.exists(os.path.join(self.path, file_path)):
                self.fnames[ft] = file_path
            else:
                missing = True

        if missing:
            # Some files are missing
            if debug:
                for ft, v in self.fnames.items():
                    assert ft_is_webp(
                        ft
                    ), "Page::__init__() : non-webp file in partial download : {}".format(
                        v
                    )
            self.skip = True

    def load(self, debug: bool = True, *args):
        if self.skip:
            return
        if len(args) == 0:
            ftypes = FileType
        else:
            ftypes = args

        for ft in ftypes:
            fname = os.path.join(self.path, self.fnames[ft])
            if debug:
                print("Page::load() : loading file {}...".format(self.fnames[ft]))

            if ft_is_webp(ft):
                # Check if the file is empty
                if os.path.getsize(fname) == 0:
                    if debug:
                        print("Page::load() : {} is empty".format(self.fnames[ft]))
                    continue
                self.files[ft] = Image.open(fname)

            elif ft_is_gz(ft):
                with gzip.open(fname) as f:
                    if ft_is_json(ft):
                        self.files[ft] = json.load(f)
                    else:
                        self.files[ft] = f.read()
            else:
                with open(fname, encoding="utf-8", errors="replace") as f:
                    if ft_is_json(ft):
                        self.files[ft] = json.load(f)
                    else:
                        self.files[ft] = f.read()


class PageLoader:
    """
    A class which loads multiple pages (with different resolutions). Loads only
    labels of the specified class
    """

    def __init__(self, path, debug: bool = True, df_cf=None, *args):
        self.path = path
        self.page_id = np.int64(os.path.basename(path))

        self.skip = False

        if debug:
            print("PageLoader()::__init__() : opening", self.page_id)

        prefixes = map(lambda x: "-".join(x.split("-")[:-1]), os.listdir(self.path))
        screen_types = list(filter(lambda x: x.find("screenshot") == -1, list(set(prefixes))))

        self.pages = dict()
        for s in screen_types:
            page = Page(self.path, s, debug)
            if not page.skip:
                self.pages[s] = page
                self.pages[s].load(debug, *args)

        self.best = next(
            (
                p
                for p in sorted(self.pages.values(), reverse=True, key=lambda p: p.width)
                if p.files.get(FileType.ScreenFull)
            ),
            None,
        )

        try:
            self.screen_type = self.best.screen_type
        except AttributeError:
            self.screen_type = ""

        if df_cf is not None:
            try:
                self.label = df_cf.loc[
                    (self.page_id, self.best.screen_type + "-screenshot.webp"),
                    "label_max",
                ]
            except (AttributeError, KeyError):
                self.label = ""
            try:
                self.certainty = df_cf.loc[
                    (self.page_id, self.best.screen_type + "-screenshot.webp"),
                    "certainty",
                ]
            except (AttributeError, KeyError):
                self.certainty = 0.0
        else:
            self.label = ""
            self.certainty = 0.0

    def image(self) -> Image.Image | None:
        """
        Find and return the largest page image width. Desktop images always have
        the largest priority
        """
        if self.best and self.best.files.get(FileType.ScreenFull) is not None:
            return self.best.files[FileType.ScreenFull]
        return None
