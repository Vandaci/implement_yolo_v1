from PIL import Image, ImageFont, ImageDraw, ImageColor
import torch
from typing import Optional, List, Union, Tuple
import warnings
import numpy as np


def drawbndbox(
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[List[str]] = None,
        colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
        fill: Optional[bool] = False,
        width: int = 10,
        font: Optional[str] = None,
        font_size: Optional[int] = None,
) -> torch.Tensor:
    '''
    Args:
        image:
        boxes: Tensor shape(N,4) [num_boxes,xmin,ymin,xmax,ymax]
        labels:
        colors:
        fill:
        width:
        font:
        font_size:

    Returns:

    '''
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"输入类型为Tensor, 非{type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor数据类型为uint8, 非{image.dtype}")
    elif image.dim() != 3:
        raise ValueError("传入单张图片, 而不是批量")
    elif image.size(0) not in {1, 3}:
        raise ValueError("只支持灰度图和RGB彩色图像")
    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("输入boxes不包含任何boxes，绘制未成功")
        return image
    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    if colors is None:
        colors = _generate_color_palette(num_boxes)
    elif isinstance(colors, list):
        if len(colors) < num_boxes:
            raise ValueError(f"Number of colors ({len(colors)}) is less than number of boxes ({num_boxes}). ")
    else:  # colors specifies a single color for all boxes
        colors = [colors] * num_boxes
    # .getrgb(color:str) 从字符串('red' etc.)转化为数值
    colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]

    if font is None:
        if font_size is not None:
            warnings.warn("字体未设置，字体大小会被忽略")
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 10, encoding='utf-8')

    # Handle Grayscale images
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        if fill:
            fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            margin = width + 1
            draw.text((bbox[0] + margin, bbox[1] + margin), label, fill=color, font=txt_font, anchor='ls')

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


def _generate_color_palette(num_objects: int):
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    return [tuple((i * palette) % 255) for i in range(num_objects)]


if __name__ == "__main__":
    img = Image.open('../data/2007_004380.jpg', 'r')
    img = np.array(img).transpose([2, 0, 1])
    img = torch.from_numpy(img)
    bndbox = torch.tensor([[8, 74, 333, 500], [5, 26, 324, 445], [40, 104, 273, 322]])
    labels = ['motorbike', 'person', 'person']
    box_img = drawbndbox(img, bndbox, labels, 'red', font='../data/msyh.ttc', font_size=12)
    box_img = box_img.permute([1, 2, 0])
    import matplotlib.pyplot as plt

    plt.imshow(box_img)
    plt.show()
    pass
