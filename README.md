# NEURAL-STYLE-TRANSFER

COMPANY NAME : CODTECH IT SOLUTIONS

NAME : KOSIREDDY RISHIKA

INTERN ID : CODF68

DOMAIN : ARTIFICIAL INTELLIGENCE

DURATION : 4WEEKS

MENTOR : NEELA SANTOSH

SMALL DISCRIPTION OF THE NEURAL STYLE TRANSFER :

This Streamlit application performs Neural Style Transfer using a pretrained VGG19 model from PyTorch. The user uploads a content image (such as a regular photograph) and a style image (such as a painting or artistic texture). The app preprocesses these images by resizing and normalizing them to match the input requirements of the VGG19 model. Then, it builds a custom model by extracting layers from VGG19 and inserting special modules that compute content loss and style loss. Content loss ensures that the generated image preserves the important structures of the content image, while style loss ensures that the textures and colors resemble those of the style image. The image generation is treated as an optimization problem, where the input image pixels are adjusted using the LBFGS optimizer to minimize a weighted combination of content and style losses. After several optimization steps, the final stylized image is displayed alongside the original content and style images. This project uses key libraries such as Streamlit for the interface, PyTorch and Torchvision for model handling, PIL for image loading, and NumPy for tensor operations. The entire app can be easily installed by setting up the environment with a few dependencies and can be launched with a single streamlit run command.

OUTPUT :

![Image](https://github.com/user-attachments/assets/e6f6bf00-1223-4c2f-8775-2a56d95d0ae7)

![Image](https://github.com/user-attachments/assets/8e012791-b379-49e8-b15f-913b20e62043)
