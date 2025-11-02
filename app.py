import streamlit as st
from pickle import load
import os
import torch
import model_data_setup
from PIL import Image

def predict(model, pretrained_transforms, image):
    ''' check again from https://www.learnpytorch.io/06_pytorch_transfer_learning/'''
    model.to(device)
    transformed_image = pretrained_transforms(image).unsqueeze(dim=0)
    model.eval()
    outputs = model(transformed_image.to(device))
    _, y_preds = torch.max(outputs, 1)
    return y_preds



#---- MAIN ------
st.set_page_config(
    page_title='Wing Classification App',
    page_icon = '🦗',
    layout = 'centered',
    menu_items = {
        "About": 'This application is a part of the "Strengthening UK-Thai Research Capacity" project funded by the Academy of Medical Sciences.'
    }
)

st.title("Biting Midges Wing Classification")

# -- load model
class_names = ['guttifer', 'peregrinus']  # these should be from class_names = list(le.classes_)
no_of_classes = len(class_names)
# get predict class
frozen_base = True  # this must be the same as the training model
model_name = 'EfficientNet_B0'
model_path = os.path.join('models/', model_name + '.pth')

# add progress bar !!!

# create model, get pretrained_transform from the model
model, pretrained_transforms, criterion, optimizer, device = model_data_setup.model_setup(model_name,
                                                                                          no_of_classes,
                                                                                          frozen_base=frozen_base)
# load best model state saved from the training process
model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))

with st.container():#st.form('main_form'):
    uploaded_image = st.file_uploader(label="Upload an image (recommended image resolution is 491x368):", key='uploaded_image', type=['tif'], accept_multiple_files=False)
    img_container = st.container(border=True)#,horizontal=True, horizontal_alignment="right")
    if uploaded_image is not None:
        # show image
        img_container.image(image=uploaded_image)
        # begin prediction process
        image = Image.open((uploaded_image))

        # resize the image!
        # for resizing an image
        # size of training images 491x368
        #resize_ratio = 10
        #image = image.resize((int(image.width / resize_ratio), int(image.height / resize_ratio)), Image.LANCZOS)

        # predict the class of a new image
        # add progress bar !!!
        with st.spinner("Prediction progress, please wait ..."):#, show_time=True):
            y_preds = predict(model, pretrained_transforms, image)
        with st.container():
            st.subheader('This insect is in the "' + class_names[y_preds] + '" species.')
            #st.write('Predicted by '+model_name)
