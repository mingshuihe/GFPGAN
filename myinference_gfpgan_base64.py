import argparse
import cv2
import numpy
import os
import torch
from gfpgan import GFPGANer
from fastapi import FastAPI, UploadFile
import uvicorn
import base64
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# 上传图片,处理之后返回base64图片编码

app = FastAPI()
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, default='inputs/whole_imgs',help='Input image or folder. Default: inputs/whole_imgs')
parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')
parser.add_argument('-v', '--version', type=str, default='1.3',help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
parser.add_argument('-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
parser.add_argument('--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
parser.add_argument('--bg_tile', type=int, default=400,help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
parser.add_argument('--ext', type=str, default='auto',help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')
args = parser.parse_args()

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
bg_upsampler = RealESRGANer(
    scale=2,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    model=model,
    tile=args.bg_tile,
    tile_pad=10,
    pre_pad=0,
    half=True)  # need to set False in CPU mode

if args.version == '1.3':
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.3'
elif args.version == '1.4':
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.4'
elif args.version == 'RestoreFormer':
    arch = 'RestoreFormer'
    channel_multiplier = 2
    model_name = 'RestoreFormer'
else:
    raise ValueError(f'Wrong model version {args.version}.')

model_path = os.path.join('gfpgan/weights', model_name + '.pth')

restorer = GFPGANer(
    model_path=model_path,
    upscale=args.upscale,
    arch=arch,
    channel_multiplier=channel_multiplier,
    bg_upsampler=bg_upsampler)

def main(input_img):
    torch.no_grad()
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=args.aligned,
        only_center_face=args.only_center_face,
        paste_back=True,
        weight=args.weight)

    is_ok,encoded_img = cv2.imencode('.PNG', restored_img)
    r_img = base64.b64encode(encoded_img)
    return r_img;

# http://localhost/post
@app.post("/post")
async def getApi(uploaded_file: UploadFile):
    contents = await uploaded_file.read()
    img = cv2.imdecode(numpy.fromstring(contents, numpy.uint8), cv2.IMREAD_UNCHANGED)
    return_image = main(img)
    return {"info":return_image}

if __name__ == '__main__':
    uvicorn.run(app, port=80)
