import torch
from matplotlib import pyplot as plt
from model import Generator
import config


def generate_and_save(model, idx):
    fake_image = model.generate()  # [1, 1, 28, 28]
    img = fake_image[0, 0].detach().cpu()  # remove batch & channel dims
    img = (img + 1) / 2  # scale from [-1,1] to [0,1]

    filename = f"generated_image_{idx}.png"
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def main():
    model = Generator()
    model.to(config.DEVICE)
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {config.MODEL_PATH}")
    model.load_state_dict(torch.load(config.MODEL_PATH))
    image_counter = 0

    while True:
        user_input = input("How many images to generate? (type 'exit' or 'cancel' to quit): ").strip()
        if user_input.lower() in ['exit', 'cancel']:
            print("Exiting program.")
            break

        if not user_input.isdigit():
            print("Please enter a valid number or 'exit'/'cancel'.")
            continue

        num_images = int(user_input)
        for _ in range(num_images):
            generate_and_save(model, image_counter)
            image_counter += 1

if __name__ == '__main__':
    main()