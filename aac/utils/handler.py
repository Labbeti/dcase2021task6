
import os
import os.path as osp

from logging import FileHandler
from typing import Optional


class FileHandlerCustom(FileHandler):
	"""FileHandler that build intermediate directories.
	Used for export hydra logs to a file contained in a folder that does not exists yet at the start of the program.
	"""
	def __init__(self, filename: str, mode: str = 'a', encoding: Optional[str] = None, delay: bool = False):
		dpath_parent = osp.dirname(filename)
		if dpath_parent != '' and not osp.isdir(dpath_parent):
			os.makedirs(dpath_parent)
		super().__init__(filename, mode, encoding, delay)
