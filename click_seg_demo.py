import sys
from pathlib import Path

import cv2
import torch
import numpy as np
from collections import namedtuple
import tkinter as tk
from tkinter import filedialog
import argparse

sys.path.insert(0, '.')
from clickseg.isegm.inference import utils
from clickseg.isegm.inference.predictors import get_predictor
from clickseg.isegm.inference.clicker import Clicker, Click

from PIL import Image, ImageTk, ImageDraw
from typing import Union
from time import time


np.int = int

def parse_args():
    parser = argparse.ArgumentParser(description='ClickSeg Demo')
    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='Path to the model checkpoint')
    parser.add_argument('--score-thr',
                        type=float,
                        default=0.5,
                        help='Threshold for the prediction score')
    return parser.parse_args()

class FCModel:
    def __init__(self,
                 score_thr: float = 0.5,
                 checkpoint_path: Union[Path, str] = None):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        checkpoint_path = checkpoint_path or Path(
            'experiments/ckpts/hr32s2-comb.pth')

        model = utils.load_is_model(checkpoint_path, device)
        self.score_thr = score_thr

        predictor_params = dict(net_clicks_limit=20, )
        zoom_in_params = dict(
            target_size=600,
            expansion_ratio=1.4,
        )

        self.predictor = get_predictor(model,
                                       'FocalClick',
                                       device,
                                       infer_size=256,
                                       prob_thresh=0.5,
                                       predictor_params=predictor_params,
                                       focus_crop_r=1.5,
                                       zoom_in_params=zoom_in_params)
        self.bool_mask = None
        self.image = None
        self.clicker = Clicker()

    def reset(self, image=None, mask=None):
        if image is not None:
            self.bool_mask = mask if mask is not None else np.zeros(
                shape=image[:2])
            self.predictor.set_input_image(image.copy())
            self.predictor.set_prev_mask(self.bool_mask > 0)
            self.clicker = Clicker()

    def set_prev_mask(self, mask=None):
        assert mask is not None
        self.bool_mask = mask > 0
        self.predictor.set_prev_mask(self.bool_mask)

    def add_click(self, coords, is_positive=None):
        x1, y1 = coords
        is_positive = is_positive or not self.bool_mask[y1, x1]
        self.clicker.add_click(Click(is_positive=is_positive, coords=(y1, x1)))

    def predict(self):
        pred_probs = self.predictor.get_prediction(self.clicker)
        self.bool_mask = pred_probs > self.score_thr
        return 255 * self.bool_mask.astype(np.uint8)

    @property
    def click_list(self):
        return self.clicker.get_clicks()


class ClickSEGUI:
    def __init__(self, main_frame, checkpoint, score_thr):
        
        self.fc_predictor = FCModel(score_thr=score_thr, checkpoint_path=checkpoint)
        self.default_length = 800
        self.control_frame_length = 150
        self.tool_frame_length = 120
        self.wh_ratio = None

        self.initiate_image_layers()
        self.initiate_ui(main_frame)
        self.main_frame.bind('<Motion>', self.on_move)
        self.mask_color = (0, 0, 139, 128)
        self.bg_color = (0, 0, 0, 0)
        self.edit_color = (0, 0, 0, 128)

    def initiate_image_layers(self):
        self.image = None
        self.mask_layer = None

        self.edit_layer = None
        self.focus_region_is_set = False
        self.scaled_focus_bbox = None

        self.draw_object = None

        self.on_click_start = None
        self.temp_rectangle = None

        self.temp_brush = None

    def init_canvas(self, canvas_frame):
        canvas_width = self.main_frame.winfo_reqwidth(
        ) - self.control_frame_length
        canvas_height = self.main_frame.winfo_reqheight(
        ) - self.tool_frame_length
        canvas_frame.config(width=canvas_width,
                            height=canvas_height,
                            padx=10,
                            pady=10)
        self.canvas = tk.Canvas(
            canvas_frame,
            bg='white',
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind('<B1-Motion>', self.draw)
        # Bind mouse button down / release events
        self.canvas.bind('<Button-1>', self.on_click_down)
        self.canvas.bind('<ButtonRelease-1>', self.on_click_release)

    def init_control_frame(self, control_frame) -> int:

        self.open_button = tk.Button(control_frame,
                                     text='打開圖像',
                                     command=self.open_image)
        self.open_button.pack()

        self.load_mask_button = tk.Button(control_frame,
                                          text='加載遮罩',
                                          command=self.load_mask)
        self.load_mask_button.pack()

        self.save_button = tk.Button(control_frame,
                                     text='保存圖像',
                                     command=self.save_image)
        self.save_button.pack()

        self.save_binary_button = tk.Button(control_frame,
                                            text='保存二元圖像',
                                            command=self.save_binary_image)
        self.save_binary_button.pack()

        self.clear_mask_button = tk.Button(control_frame,
                                           text='清除遮罩',
                                           command=self.clear_mask)
        self.clear_mask_button.pack()

        self.reset_model_button = tk.Button(control_frame,
                                            text='重設模型',
                                            command=self.reset_model)
        self.reset_model_button.pack()

        self.set_focus_region_mode = tk.BooleanVar()
        self.set_focus_region_checkbox = tk.Checkbutton(
            control_frame,
            text='選擇 ROI',
            variable=self.set_focus_region_mode,
            command=self.set_focus_region)
        self.set_focus_region_checkbox.pack(pady=(20, 0))

        self.show_focus_region_mode = tk.BooleanVar()
        self.show_focus_region_checkbox = tk.Checkbutton(
            control_frame,
            text='只顯示 ROI',
            variable=self.show_focus_region_mode,
            command=self.display_image)
        self.show_focus_region_checkbox.pack()

    # def show_focus_region(self):
    #     self.display_image()

    def set_focus_region(self):
        if self.set_focus_region_mode.get():
            self.draw_mode.set('interactive')
        else:
            self.edit_layer = None
            self.focus_region_is_set = False
            self.scaled_focus_bbox = None
            self.show_focus_region_mode.set(False)
            self.display_image()
            self.reset_model()

    def init_display_frame(self, display_mode_frame):

        self.display_image_mode = tk.BooleanVar(value=True)
        self.display_mask_mode = tk.BooleanVar(value=True)

        self.display_image_checkbox = tk.Checkbutton(
            display_mode_frame,
            text='顯示圖像',
            variable=self.display_image_mode,
            command=self.display_image)
        self.display_image_checkbox.grid(row=0, column=0)

        self.display_mask_checkbox = tk.Checkbutton(
            display_mode_frame,
            text='顯示遮罩',
            variable=self.display_mask_mode,
            command=self.display_image)
        self.display_mask_checkbox.grid(row=0, column=1)

    def init_tool_frame(self, tool_frame):

        self.brush_size_label = tk.Label(tool_frame, text='筆刷大小：')
        self.brush_size_label.grid(row=0, column=5)

        self.brush_size_scale = tk.Scale(tool_frame,
                                         from_=5,
                                         to=70,
                                         orient=tk.HORIZONTAL,
                                         variable=self.brush_size)
        self.brush_size_scale.grid(row=0, column=6)

        self.draw_mode_label = tk.Label(tool_frame, text='模式：')
        self.draw_mode_label.grid(row=0, column=0)

        self.draw_mode_brush = tk.Radiobutton(tool_frame,
                                              text='筆刷',
                                              variable=self.draw_mode,
                                              value='brush')

        self.draw_mode_brush.grid(row=0, column=1)

        self.draw_mode_rectangle = tk.Radiobutton(tool_frame,
                                                  text='方框',
                                                  variable=self.draw_mode,
                                                  value='rectangle')

        self.draw_mode_rectangle.grid(row=0, column=2)

        self.draw_mode_interactive_tool = tk.Radiobutton(
            tool_frame,
            text='啟動點擊互動工具',
            variable=self.draw_mode,
            value='interactive')

        self.draw_mode_interactive_tool.grid(row=0, column=3)

        self.clear_mode_var = tk.BooleanVar()
        self.clear_mode_checkbox = tk.Checkbutton(tool_frame,
                                                  text='清除模式',
                                                  variable=self.clear_mode_var)
        self.clear_mode_checkbox.grid(row=0, column=4)

    def initiate_ui(self, main_frame):
        self.main_frame = main_frame
        self.main_frame.title('圖像塗改工具')

        self.brush_size = tk.IntVar()
        self.brush_size.set(20)

        self.draw_mode = tk.StringVar()
        self.draw_mode.set('brush')

        # 使用 Frame 容納畫布和按鈕
        self.upper_frame = tk.Frame(self.main_frame)
        self.upper_frame.grid(row=0, column=0, padx=10, pady=10)

        self.left_frame = tk.Frame(self.upper_frame)
        self.left_frame.grid(row=0, column=0)

        self.right_frame = tk.Frame(self.upper_frame)
        self.right_frame.grid(row=0, column=1, padx=10)

        self.display_mode_frame = tk.Frame(self.main_frame)
        self.display_mode_frame.grid(row=1, column=0, padx=10, sticky=tk.W)

        self.tool_frame = tk.Frame(self.main_frame)
        self.tool_frame.grid(row=2, column=0, padx=10, sticky=tk.W)

        self.status_text = tk.StringVar(value='座標：')
        self.status = tk.Label(self.main_frame, textvariable=self.status_text)
        self.status.grid(row=3, column=0, padx=10, sticky=tk.W)

        self.init_control_frame(self.right_frame)
        self.init_display_frame(self.display_mode_frame)
        self.init_tool_frame(self.tool_frame)
        self.init_canvas(self.left_frame)

        self.main_frame.bind('<Configure>', self.on_resize)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[('Image files',
                                                           ('*.png', '*.jpeg',
                                                            '*.jpg',
                                                            '*.bmp'))])
        if file_path:
            self.image = Image.open(file_path).convert('RGBA')
            self.wh_ratio = self.image.width / self.image.height
            self.set_mask(
                rgba_mask=Image.new('RGBA', self.image.size, (0, 0, 0, 0)))

            self.to_resize_layout(mode='on_open')
            self.reset_model()

    def reset_model(self):
        if self.image:
            if self.focus_region_is_set:
                p1, q1, p2, q2 = self.scaled_focus_bbox
                image = self.get_image_array()[q1:q2, p1:p2, :]
                mask = self.get_binary_mask()[q1:q2, p1:p2]
            else:
                image = self.get_image_array()
                mask = self.get_binary_mask()
            self.fc_predictor.reset(image=image, mask=mask)
            # to reset clicks
            self.display_image()

    def load_mask(self):
        if self.image:
            mask_path = filedialog.askopenfilename(filetypes=[('Image files',
                                                               ('*.png',
                                                                '*.jpeg',
                                                                '*.jpg',
                                                                '*.bmp'))])
            if not mask_path or not Path(mask_path).exists():
                return

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # bisect: 0 ~ 127 -> 0 & 128 ~ 255 -> 255
            _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

            self.set_mask(binary_mask=mask)

    def set_mask(self, binary_mask=None, rgba_mask=None):
        assert (binary_mask is not None) ^ (rgba_mask is not None)
        rgba_mask = self.mask_to_pil_image(
            binary_mask) if binary_mask is not None else rgba_mask
        self.mask_layer = rgba_mask
        self.draw_object = ImageDraw.Draw(rgba_mask)

    def save_image(self):
        if self.image and self.mask_layer:
            merged_image = Image.alpha_composite(self.image, self.mask_layer)
            file_path = filedialog.asksaveasfilename(defaultextension='.jpg')
            if file_path:
                if Path(file_path).suffix in [
                        '.jpg', '.jpeg', '.JPG', '.JPEG'
                ]:
                    merged_image.convert('RGB').save(
                        file_path,
                        format='JPEG',
                        quality=self.get_jpeg_quality())
                else:
                    merged_image.save(file_path)

    def save_binary_image(self):
        if self.mask_layer:
            binary_image = self.get_binary_mask()
            binary_image = np.concatenate(
                [binary_image[:, :, None] for _ in range(3)], axis=-1)
            file_path = filedialog.asksaveasfilename(defaultextension='.png')
            if file_path:
                cv2.imwrite(file_path, binary_image)

    def get_jpeg_quality(self):
        if self.image.format == 'JPEG':
            return self.image.info.get('quality', 95)
        return 95

    def get_binary_mask(self):
        binary_mask = self.mask_layer.copy()
        # self.mask_color -> self.mask_color[-1] -> 255
        binary_mask = np.array(binary_mask, dtype=np.uint8)[:, :, -1]
        binary_mask[binary_mask == self.mask_color[-1]] = 255
        return binary_mask

    def get_image_array(self):
        image_array = self.image.copy().convert('RGB')
        image_array = np.array(image_array, dtype=np.uint8)
        return image_array

    def display_image(self):
        if self.image and self.mask_layer:

            is_display_image = self.display_image_mode.get()
            is_display_mask = self.display_mask_mode.get()

            if is_display_mask and self.edit_layer is not None:
                merge_mask = Image.alpha_composite(self.edit_layer,
                                                   self.mask_layer)
            else:
                merge_mask = self.mask_layer

            if is_display_image and is_display_mask:
                merged_image = Image.alpha_composite(self.image, merge_mask)
            elif is_display_image:
                merged_image = self.image
            elif is_display_mask:
                merged_image = merge_mask
            else:
                merged_image = Image.new('RGBA', self.image.size, (0, 0, 0, 0))

            if is_display_mask:
                for click in self.fc_predictor.click_list:
                    q, p = click.coords
                    if self.scaled_focus_bbox is not None:
                        p1, q1 = self.scaled_focus_bbox[:2]
                        p, q = p + p1, q + q1
                    ImageDraw.Draw(merged_image).ellipse(
                        [p - 10, q - 10, p + 10, q + 10],
                        fill='green' if click.is_positive else 'red',
                        outline='black',
                        width=2)

            if self.show_focus_region_mode.get():
                merged_image = merged_image.crop(self.scaled_focus_bbox)
                self.crop_scale_x = merged_image.width / self.canvas.winfo_width(
                )
                self.crop_scale_y = merged_image.height / self.canvas.winfo_height(
                )

            merged_image_resized = merged_image.resize(
                (self.canvas.winfo_width(), self.canvas.winfo_height()),
                Image.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(merged_image_resized)
            self.canvas.delete('image')
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

    def to_resize_layout(self, mode=None):
        assert mode, 'mode must be specified'
        if mode == 'on_resize':
            canvas_width = self.main_frame.winfo_width(
            ) - self.control_frame_length
            canvas_height = self.main_frame.winfo_height(
            ) - self.tool_frame_length

            self.default_length = max(self.main_frame.winfo_width(),
                                      self.main_frame.winfo_height())

        elif mode == 'on_open':
            wh_ratio = self.wh_ratio or 1.6
            canvas_width = (
                self.default_length -
                self.control_frame_length) if wh_ratio > 1 else (
                    self.default_length - self.tool_frame_length) * wh_ratio
            canvas_height = canvas_width / wh_ratio
            whole_width = canvas_width + self.control_frame_length
            whole_height = canvas_height + self.tool_frame_length
            self.main_frame.geometry(f'{int(whole_width)}x{int(whole_height)}')

        self.left_frame.config(width=int(canvas_width),
                               height=int(canvas_height))

    def on_resize(self, event):
        # check this event is triggered by root window
        if isinstance(event.widget, tk.Label):
            return False
        # print('display triggered by', type(event.widget))
        if self.image:
            self.scale_x = self.image.width / self.canvas.winfo_width()
            self.scale_y = self.image.height / self.canvas.winfo_height()
        self.to_resize_layout(mode='on_resize')
        self.display_image()

    def brush(self, event):
        p, q = event.x * self.scale_x, event.y * self.scale_y
        r = self.brush_size.get()
        rp = r * self.scale_x
        rq = r * self.scale_y
        self.draw_object.ellipse(
            [p - rp, q - rq, p + rp, q + rq],
            fill=self.bg_color
            if self.clear_mode_var.get() else self.mask_color,
            width=0)

    def draw(self, event):
        if not self.image:
            return False

        if (self.set_focus_region_mode.get() or self.draw_mode.get()
                == 'rectangle') and self.on_click_start is not None:
            if self.temp_rectangle:
                self.canvas.delete(self.temp_rectangle)
            self.temp_rectangle = self.canvas.create_rectangle(
                self.on_click_start[0],
                self.on_click_start[1],
                event.x,
                event.y,
                outline='black',
                width=2)

        elif self.draw_mode.get() == 'brush':
            self.brush(event)
            self.display_image()

    def on_move(self, event):
        if self.image and isinstance(event.widget, tk.Canvas):
            x, y = event.x, event.y
            p, q = x * self.scale_x, y * self.scale_y
            # self.status.config(text=f'座標：({p:.0f}, {q:.0f})')
            self.status_text.set(f'座標：({p:.0f}, {q:.0f})')

            if self.set_focus_region_mode.get():
                pass
            elif self.draw_mode.get() == 'brush':
                if self.temp_brush:
                    self.canvas.delete(self.temp_brush)
                r = self.brush_size.get()
                if (0 < x < self.canvas.winfo_width()
                        and 0 < y < self.canvas.winfo_height()):
                    self.temp_brush = self.canvas.create_oval(
                        x - r,
                        y - r,
                        x + r,
                        y + r,
                        fill='#000080',
                        width=0,
                    )

        else:
            if self.temp_brush:
                self.canvas.delete(self.temp_brush)
            self.status_text.set(f'座標：')

    @staticmethod
    def mask_to_pil_image(mask):
        if np.ndim(mask) == 3:
            mask = mask[:, :, 0]
        rgba_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba_mask[mask == 255] = [0, 0, 139, 128]  # 塗改區域

        # 將 NumPy 陣列轉換為 Pillow Image 物件
        rgba_mask = Image.fromarray(rgba_mask, mode='RGBA')
        return rgba_mask

    def on_click_down_to_draw(self, event):
        x, y = event.x, event.y
        if self.draw_mode.get() == 'brush':
            self.brush(event)
        elif self.draw_mode.get() == 'rectangle':
            self.on_click_start = (x, y)

        elif self.draw_mode.get() == 'interactive':

            x1, y1 = x * self.scale_x, y * self.scale_y
            self.fc_predictor.set_prev_mask(self.get_binary_mask())
            self.fc_predictor.add_click((int(x1), int(y1)))
            mask = self.fc_predictor.predict()
            self.set_mask(binary_mask=mask)
            self.display_image()

    def on_click_down(self, event):
        x, y = event.x, event.y

        if not self.image:
            return False

        if self.set_focus_region_mode.get():
            if not self.focus_region_is_set:
                self.on_click_start = (x, y)
            elif self.draw_mode.get() == 'interactive':
                p1, q1, p2, q2 = self.scaled_focus_bbox
                mask = self.get_binary_mask()
                cropped_mask = mask[q1:q2, p1:p2]

                if self.show_focus_region_mode.get():
                    p, q = int(x * self.crop_scale_x), int(y *
                                                           self.crop_scale_y)
                else:
                    p, q = int(x * self.scale_x) - p1, int(
                        y * self.scale_y) - q1

                # print(f'show_focus_region_mode: {self.show_focus_region_mode.get()}; focus bbox: {self.scaled_focus_bbox}, p, q = {(p,q)}')
                if not 0 <= p < p2 - p1 and 0 <= q < q2 - q1:
                    return False

                self.fc_predictor.set_prev_mask(cropped_mask)
                self.fc_predictor.add_click((p, q))
                cropped_mask = self.fc_predictor.predict()
                mask[q1:q2, p1:p2] = cropped_mask
                self.set_mask(binary_mask=mask)
                self.display_image()

        else:
            self.on_click_down_to_draw(event=event)

    def on_click_release(self, event):
        if not self.image:
            return False
        if self.on_click_start:
            p2, q2 = event.x * self.scale_x, event.y * self.scale_y
            p1, q1 = self.on_click_start[
                0] * self.scale_x, self.on_click_start[1] * self.scale_y
            p1, p2 = sorted([p1, p2])
            q1, q2 = sorted([q1, q2])
            p1, q1, p2, q2 = [
                int(i) for i in [p1, q1, np.ceil(p2),
                                 np.ceil(q2)]
            ]
            self.on_click_start = None
            if self.set_focus_region_mode.get(
            ) and not self.focus_region_is_set:
                self.edit_layer = Image.new('RGBA', self.image.size,
                                            (0, 0, 0, 128))
                self.focus_draw = ImageDraw.Draw(self.edit_layer)
                self.focus_draw.rectangle([p1, q1 - 1, p2, q2 - 1],
                                          fill=self.bg_color,
                                          width=0)
                self.scaled_focus_bbox = [p1, q1, p2, q2]
                self.focus_region_is_set = True

                self.reset_model()

            elif self.draw_mode.get() == 'rectangle':
                self.draw_object.rectangle(
                    [p1, q1, p2, q2],
                    fill=self.bg_color
                    if self.clear_mode_var.get() else self.mask_color,
                    width=0)

            if self.temp_rectangle:
                self.canvas.delete(self.temp_rectangle)

            self.display_image()

    def clear_mask(self):
        if self.mask_layer:
            self.set_mask(
                rgba_mask=Image.new('RGBA', self.image.size, (0, 0, 0, 0)))
            self.display_image()


if __name__ == '__main__':
    
    args = parse_args()
    checkpoint = args.checkpoint
    score_thr = args.score_thr
    
    assert Path(checkpoint).exists(), f'checkpoint not found: {checkpoint}'
    
    root = tk.Tk()

    # root.geometry('800x600+0+0') # width x height + x_offset + y_offset
    root.geometry('800x600')  # width x height + x_offset + y_offset
    root.minsize(400, 400)
    editor = ClickSEGUI(root, checkpoint, score_thr)

    root.mainloop()