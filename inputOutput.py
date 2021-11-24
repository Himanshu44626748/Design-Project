from examples.ex_fasa_saliency_map.ex_fasa_saliency_map_images import main
from examples.ex_skin_detection_images.ex_skin_detection_images import skin_detection
from examples.ex_color_detection_image.ex_color_detection_image import color_detection
from gbvs import demo, ittikochneibur, gbvs
import os
import cv2

print("Enter 1 to run FASA algorithm")
print("Enter 2 to run Skin Detection algorithm")
print("Enter 3 to run Color Detection algorithm")
print("Enter 4 to run Graph Based Visual Saliency algorithm")
choice = int(input())

def load_images_from_folder(folder):
    images_path = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images_path.append(folder+"/"+filename)

    return  images_path


if choice == 1:
    print("Note: Image must have height = 400px and width = 400px")
    folder = input("Enter the path of folder which contains images\n")

    images_path = load_images_from_folder(folder)

    if __name__ == '__main__':
        main(images_path)

elif choice == 2:
    print("Note: Image must have height = 1080px and width = 1920px")
    folder = input("Enter the path of folder which contains images\n")

    images_path = load_images_from_folder(folder)

    skin_detection(images_path)

elif choice == 3:
    folder = input("Enter the path of folder which contains images\n")

    images_path = load_images_from_folder(folder)

    color_detection(images_path)

elif choice == 4:
    folder = input("Enter the path of folder which contains images\n")

    images_path = load_images_from_folder(folder)

    demo.gbvs_ittiKoch_salience(images_path)


