import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


if __name__ == '__main__':
    import cv2
    processor = BlipProcessor.from_pretrained('/home/xmuairmud/jyx/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('/home/xmuairmud/jyx/blip-image-captioning-base', torch_dtype=torch.float16).cuda()
    image1 = Image.open('/home/xmuairmud/jyx/data/GTA/images/images/00944.png')
    inputs = processor(images=image1, text='a picture of cityscapes')
    output = model.generate(**inputs)
    print(output)
    # image1 = cv2.imread('/home/xmuairmud/jyx/data/GTA/images/images/24965.png')
    # image1 = cv2.resize(image1, dsize=(384, 384))
    # image1 = torch.Tensor(image1).permute((2, 0, 1)).cuda()[None, ...]
    # image2 = cv2.imread('/home/xmuairmud/jyx/data/GTA/images/images/24966.png')
    # image2 = cv2.resize(image2, dsize=(384, 384))
    # image2 = torch.Tensor(image2).permute((2, 0, 1)).cuda()[None, ...]

    # images = torch.cat((image1, image2), dim=0)

    # inputs = processor(images=images, text='a picture of cityscapes')
    # print(len(inputs['pixel_values']), len(inputs['pixel']) inputs['pixel_values'][0].shape)
    
    # output = model.generate(**inputs)
    # print(output)


    # from lavis.models import load_model_and_preprocess
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # # this also loads the associated image processors
    # model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    # # preprocess the image
    # # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    # image1 = Image.open('/home/xmuairmud/jyx/data/GTA/images/images/24965.png')
    # image2 = Image.open('/home/xmuairmud/jyx/data/GTA/images/images/00944.png')
    # image = vis_processors["eval"](image2).unsqueeze(0).to(device)
    # # generate caption
    # output = model.generate({"image": image})
    # print(output)
