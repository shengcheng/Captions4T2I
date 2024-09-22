from PIL import Image
from io import BytesIO
import numpy as np
import random

import os
import json
import base64
from tqdm import tqdm
from typing import Optional, List, Dict, TypedDict, Union

DATASET_BASE = "data/densely_captioned_images"
DATASET_COMPLETE_PATH = os.path.join(DATASET_BASE, "complete")
DATASET_PHOTO_PATH = ""

from transformers import CLIPProcessor

# Utilities

processor = None
def get_clip_processor() -> CLIPProcessor:
    global processor
    if processor is None:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return processor

def get_clip_token_length(in_str: str) -> int:
    processor = get_clip_processor()
    return len(processor.tokenizer(in_str)['input_ids'])


class DCIEntry(TypedDict):
    """Generated example returned from the DCI for training"""
    image: np.ndarray
    caption: str
    key: str


class PointDict(TypedDict):
    """Simple dict representing a point"""
    x: int
    y: int


class BoundDict(TypedDict):
    """Dict for a bounding box"""
    topLeft: PointDict
    bottomRight: PointDict


class NegativeEntry(TypedDict):
    """Dict containing negatives"""
    swaps: List[str]
    layout: List[str]
    basic: List[str]


DCISummaries = Dict[str, str]
DCINegatives = Dict[str, NegativeEntry]


class MaskDataEntry(TypedDict):
    """Mask-associated data dict"""
    outer_mask: str # base64 string representation for the actual pixel mask
    area: int # area in pixels of the mask
    bounds: BoundDict # Bounding box for mask
    idx: int # int index into the dict of masks
    requirements: List[int] # List of masks that this one contains
    parent: int # index of containing mask, -1 if main image
    label: str # text label for this mask
    caption: str # caption for this mask
    mask_quality: int # one of 0 (fine) 1 (low_quality) or 2 (unusable)


class DCIBaseData(TypedDict):
    """
    Class of expected fields we should find in the stored json files
    """
    short_caption: str # the standard caption
    extra_caption: str # additional description not captured elsewhere
    image: str # path to the image
    mask_data: Dict[str, MaskDataEntry]
    mask_keys: List[str] # list of available masks in the image


class DCIExtendedData(DCIBaseData):
    """
    Class including additional data we cache or prepare for a DCI
    """
    img: np.ndarray # actual image in np
    height: int # image height
    _id: int # idx of this datapoint
    _entry_key: str # actual computed file key 
    _source: str # full filepath of source


ENTRIES = None
ENTRIES_MAP = None
ENTRIES_REVERSE_MAP = None


def init_entries(source: str = DATASET_COMPLETE_PATH) -> None:
    """
    Initalized the globally loaded entries from the source
    for easier access by other functions
    """
    global ENTRIES, ENTRIES_MAP, ENTRIES_REVERSE_MAP
    ENTRIES = os.listdir(source)
    ENTRIES_MAP = {str(i): e for i, e in enumerate(ENTRIES)}
    ENTRIES_REVERSE_MAP = {str(e): i for i, e in enumerate(ENTRIES)}


def get_dci_count() -> int:
    """
    Return the number of available DCIs loaded into the entry map
    """
    if ENTRIES_MAP is None:
        init_entries()
    return len(ENTRIES_MAP)


def get_key_for(idx: Union[int, str]) -> str:
    """
    Wrapper for id -> entry, helper for DenseCaptionedImage
    """
    if ENTRIES_MAP is None:
        init_entries()
    return ENTRIES_MAP.get(str(idx))

def load_image_info(entry_key: str) -> Dict:
    """
    Get the associated data for a DCI loaded from the json file.
    """
    if ENTRIES is None:
        init_entries()

    # Convert idx to key
    if entry_key in ENTRIES_MAP:
        entry_key = ENTRIES_MAP[entry_key]

    complete_path = os.path.join(DATASET_COMPLETE_PATH, entry_key)
    with open(complete_path) as entry_file:
        base_data: DCIBaseData = json.load(entry_file)
    return base_data

def load_image(entry_key: str) -> DCIExtendedData:
    """
    Get the associated data for a DCI loaded from the json file.
    """
    if ENTRIES is None:
        init_entries()

    # Convert idx to key
    if entry_key in ENTRIES_MAP:
        entry_key = ENTRIES_MAP[entry_key]

    complete_path = os.path.join(DATASET_COMPLETE_PATH, entry_key)
    with open(complete_path) as entry_file:
        base_data: DCIBaseData = json.load(entry_file)
   
    img = Image.open(os.path.join(DATASET_PHOTO_PATH, base_data['image']))
    width, height = img.size
    base_data['width'] = width
    base_data['height'] = height
    base_data['img'] = np.array(img)
    base_data['_id'] = ENTRIES_REVERSE_MAP[entry_key]
    base_data['_entry_key'] = entry_key
    base_data['_source'] = os.path.join(DATASET_COMPLETE_PATH, entry_key)
    
    return base_data


class DenseCaptionedImage():
    def __init__(self, img_id: int):
        self._data: DCIExtendedData = load_image_info(str(img_id))
        self._id: int = img_id

    ### helpers ###

    def get_summaries(self) -> Optional[DCISummaries]:
        """Return summaries, if they're available, otherwise None"""
        return self._data.get('summaries')

    def get_negatives(self) -> Optional[DCINegatives]:
        """Return negatives, if they're available, otherwise None"""
        return self._data.get('negatives')

    def get_mask(self, idx: int) -> MaskDataEntry:
        return self._data['mask_data'].get(str(idx))
    
    def get_all_masks(self) -> List[MaskDataEntry]:
        return list(self._data['mask_data'].values())

    def filter_masks_by_size(
            self, 
            min_height: int = 224, 
            min_width: int = 224, 
            min_length: int = 0, 
            max_length: Optional[int] = None
    ) -> List[MaskDataEntry]:
        all_masks = self.get_all_masks()
        if min_height == 0 and min_width == 0 and min_length == 0 and max_length is None:
            return all_masks
        def mask_is_bigger(mask: MaskDataEntry) -> bool:
            if min_length > 0 or max_length is not None:
                caption_len = get_clip_token_length(self._extract_caption(mask))
                if caption_len < min_length or (max_length is not None and caption_len > max_length):
                    return False
            crop_dims = mask['bounds']
            width = crop_dims['bottomRight']['x'] - crop_dims['topLeft']['x']
            height = crop_dims['bottomRight']['y'] - crop_dims['topLeft']['y']
            return width >= min_width and height >= min_height
        return [m for m in all_masks if mask_is_bigger(m)]

    def label_masks_by_size(
            self,
            min_height: int = 224,
            min_width: int = 224,
            min_length: int = 0,
            max_length: Optional[int] = None
    ) -> List[MaskDataEntry]:
        all_masks = self.get_all_masks()
        if min_height == 0 and min_width == 0 and min_length == 0 and max_length is None:
            return all_masks

        def mask_is_bigger(mask: MaskDataEntry) -> bool:
            if min_length > 0 or max_length is not None:
                caption_len = get_clip_token_length(self._extract_caption(mask))
                if caption_len < min_length or (max_length is not None and caption_len > max_length):
                    return False
            crop_dims = mask['bounds']
            width = crop_dims['bottomRight']['x'] - crop_dims['topLeft']['x']
            height = crop_dims['bottomRight']['y'] - crop_dims['topLeft']['y']
            return width >= min_width and height >= min_height
        results = []
        for mask in all_masks:
            mask['size_fit'] = mask_is_bigger(mask)
            results.append(mask)
        return results


    def _extract_caption(self, mask) -> Optional[str]:
        if mask['mask_quality'] == 2:
            return None
        elif mask['mask_quality'] == 1:
            return mask['label']
        return f"{mask['label']}: {mask['caption']}"
    
    def _get_max_depth(self, mask) -> int:
        submasks = [self.get_mask(m) for m in mask['requirements']]
        submasks = [m for m in submasks if m is not None]
        if len(submasks) == 0:
            return 0
        return 1 + max([self._get_max_depth(m) for m in submasks])
    
    def get_all_submasks_dfs(
            self, 
            mask: MaskDataEntry, 
            max_depth: int, 
            include_self: bool = True
    ) -> List[MaskDataEntry]:
        if include_self:
            res = [mask]
        else:
            res = []
        if max_depth == 0:
            return res
        submasks = [self.get_mask(m) for m in mask['requirements']]
        submasks = [m for m in submasks if m is not None and m['mask_quality'] != 2]
        for submask in submasks:
            res += self.get_all_submasks_dfs(submask, max_depth-1)
        return res
        
    
    def get_caption_with_subcaptions(self, mask, max_depth=999) -> List[DCIEntry]:
        submasks = self.get_all_submasks_dfs(mask, max_depth=max_depth, include_self=False)
        
        # base_caption = f"This is an image of {mask['label']}. {mask['caption']}"
        base_caption = f"{mask['caption']}"
        captions = [self._extract_caption(m) for m in submasks]
        captions = [c for c in captions if c is not None]
        all_captions = "\n".join(captions)
        if len(captions) > 0:
            caption = f"{base_caption}\nThe following can also be seen in the image:\n{all_captions}"
        else:
            caption = base_caption
        if 'size_fit' in mask.keys():
            return [
            {'short_caption': caption, 'key': f'm-{mask["idx"]}-sc', 'area':mask['area'], 'size_fit': mask['size_fit']}
        ]
        return [
            {'short_caption': caption, 'key': f'm-{mask["idx"]}-sc', 'area':mask['area']}
        ]
        
    
    ### Image loading methods
    
    def get_image(self) -> np.ndarray:
        return self._data['img']
    
    def get_subimage(self, mask, pad_amount = 0.15, apply_mask=False) -> np.ndarray:
        base_image = self.get_image()
        if apply_mask:
            mask_array = np.array(Image.open(BytesIO(base64.b64decode(mask['outer_mask']))))
            mask_tiled = np.tile(mask_array, (3,1,1)).transpose((1,2,0))
            base_image = base_image * mask_tiled + (base_image // 3 + 84) * (1-mask_tiled)
            base_image = base_image.astype(np.uint8)
        x1, y1, x2, y2 = 0, 0, self._data['width'], self._data['height']
        crop_dims = mask['bounds']
        width = crop_dims['bottomRight']['x'] - crop_dims['topLeft']['x']
        height = crop_dims['bottomRight']['y'] - crop_dims['topLeft']['y']

        w_pad_int = int(width * pad_amount)
        h_pad_int = int(height * pad_amount)
        cx1 = max(crop_dims['topLeft']['x'] - w_pad_int, x1)
        cy1 = max(crop_dims['topLeft']['y'] - h_pad_int, y1)
        cx2 = min(crop_dims['bottomRight']['x'] + w_pad_int, x2)
        cy2 = min(crop_dims['bottomRight']['y'] + h_pad_int, y2)
        return base_image[cy1:cy2, cx1:cx2]
    
    ### sample creation methods
    
    def get_base_caption(self) -> List[DCIEntry]:
        area = self._data['height'] * self._data['width']
        return [
            {'short_caption': self._data['short_caption'], 'key': 'base', 'area': area}
        ]
        
    def get_extended_caption(self) -> List[DCIEntry]:
        caption = f"{self._data['short_caption']}\n{self._data['extra_caption']}"
        area = self._data['height'] * self._data['width']
        return [
            {'caption': caption, 'key': 'extended'}
        ]
    
    def get_formatted_description_min_size(self, min_height=224, min_width=224) -> List[DCIEntry]:
        base_caption = f"{self._data['short_caption']}\n{self._data['extra_caption']}"
        masks = self.filter_masks_by_size(min_height=min_height, min_width=min_width)
        captions = [self._extract_caption(m) for m in masks]
        captions = [c for c in captions if c is not None]
        all_captions = "\n".join(captions)
        if len(captions) > 0:
            caption = f"{base_caption}\nThe following can also be seen in the image:\n{all_captions}"
        else:
            caption = base_caption
        return [
            {'caption': caption, 'key': 'base'}
        ]
    
    def get_formatted_complete_description(self) -> List[DCIEntry]:
        return self.get_formatted_description_min_size(min_height=0, min_width=0)
    
    def get_positive_mask_samples(
            self, 
            min_height: int = 224, 
            min_width: int = 224, 
            min_length: int = 0, 
            max_length: Optional[int] = None
    ) -> List[DCIEntry]:
        masks = self.filter_masks_by_size(
            min_height=min_height, min_width=min_width, min_length=min_length, max_length=max_length
        )
        return [
            {
                'image': self.get_subimage(mask), 
                'caption': self._extract_caption(mask), 
                'key': f'm-{mask["idx"]}'
            } for mask in masks
        ]

def get_summarized_dataset_with_settings(
        load_base_image: bool = True,
        load_subcaptions: bool = True,
        negative_source: str = 'swaps', # which LLM prompt to use for negs, swaps|layout|basic|any|spacy
        negative_strategy: str = 'rand', # how to select which neg of multiple options, rand|first|hardest
        count: Optional[int] = None,
) -> List[List[DCIEntry]]:
    assert negative_source in ['swaps', 'layout', 'basic', 'any', 'spacy', 'spacy-ant'], f"Bad neg source {negative_source}"
    assert negative_strategy in ['rand', 'first', 'hardest'], f"Bad neg strat {negative_strategy}"

    with open(os.path.join(DATASET_BASE, 'splits.json')) as jsonf:
        split_metadata = json.load(jsonf)
        sources = []
        for split in ["train", "valid", "test"]:
            sources += split_metadata[split]

    entries_per_image = {}
    for source_path in tqdm(sources, desc="Loading Dense Caps:"):
        if count is not None and len(entries_per_image) > count:
            break
        try:
            dci = DenseCaptionedImage(source_path)
            summaries = dci.get_summaries()
            if summaries is None:
                continue

            negatives = dci.get_negatives()
            if negatives is None or len(negatives) == 1 and load_subcaptions:
                continue

            entries = []
            if load_base_image:
                # entries += dci.get_formatted_complete_description()
                entries += dci.get_base_caption()

            
            if load_subcaptions:
                all_masks = dci.filter_masks_by_size()
                entries += [dci.get_caption_with_subcaptions(m, max_depth=0)[0] for m in all_masks]

            # Remap to summaries
            for entry in entries:
                if isinstance(summaries[entry['key']], str):
                    entry['caption'] = summaries[entry['key']]
                else:
                    entry['caption'] = summaries[entry['key']][0]
                    entry['captions'] = summaries[entry['key']]
            
            # Include negatives:
            for entry in entries:

                negs = negatives[entry['key']]
                if entry['key'] not in dci._data['clip_scores'].keys():
                    clip_score = {'sum': 0, 'basic_0': 0, 'layout_0': 0, 'swaps_0': 0}
                else:
                    clip_score = dci._data['clip_scores'][entry['key']]
                entry['score'] = clip_score
                entry['negative'] = negs
            entries += dci.get_extended_caption()
            entries_dict = {}

            for entry in entries:
                key = entry['key']
                entry.pop('key', None)
                entries_dict[key] = entry




            entries_per_image[dci._data['image']] = entries_dict
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"Skipping image {source_path} due to loading issue")

    with open("data/raw.json", 'w') as jsonf:
        json.dump(entries_per_image, jsonf)

    return entries_per_image

def get_data_iterator(source, cap=1e30):
    def gen_dci():
        yielded_count = 0
        for i, dci_entry_list in enumerate(source):
            for entry in dci_entry_list:
                yield (i, entry)
                yielded_count += 1
                if yielded_count == cap:
                    return
    return gen_dci


def get_negative_by_strategy(
        negatives: NegativeEntry,
        negative_source: str = 'swaps',  # which LLM prompt to use for negs, swaps|layout|basic|any
        negative_strategy: str = 'rand',  # how to select which neg of multiple options, rand|first|hardest
        clip_scores: Optional[Dict[str, float]] = None,
) -> str:
    negative_sources = []
    scores = []
    if negative_source != 'any':
        negative_cands = negatives[negative_source]
        negative_sources = [f"{negative_source}_{i}" for i in range(len(negative_cands))]

    else:
        negative_cands = []
        for neg_type, val in negatives.items():
            negative_cands += val
            negative_sources += [f"{neg_type}_{i}" for i in range(len(val))]



    if negative_strategy == 'first':
        return negative_cands[0]
    elif negative_strategy == 'rand':
        return random.choice(negative_cands)
    elif negative_strategy == 'hardest':
        zipped = list(zip(negative_cands, negative_sources))
        zipped.sort(key=lambda x: clip_scores[x[1]])
        return zipped[-1][0]


if __name__ == '__main__':
    get_summarized_dataset_with_settings()