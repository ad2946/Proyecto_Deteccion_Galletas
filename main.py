from detecto.core import Dataset
import cv2
from detecto import utils
from detecto.core import Model
from torchvision import transforms
from detecto.visualize import detect_video
import matplotlib.pyplot as plt
import click

@click.group()
def cli():
    pass

#Funcion sobreescrita para probar sobre detecto
def detect_live2(model,score_filter=0.6):
    errores = 0
    cv2.namedWindow('Detecto')
    try:
        video = cv2.VideoCapture(0)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        print(width)
        print(height)
    except:
        print('No webcam available.')
        return

    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.putText(frame, 'Errores:{}'.format(errores),(0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        labels, boxes, scores = model.predict(frame)

        # Plot each box with its label and score
        for i in range(boxes.shape[0]):
            if scores[i] < score_filter:
                continue

            box = boxes[i]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
            if labels:
                fail_detected=False
                if 'Galleta_rota' in labels:
                    fail_box_min=int(box[0])
                    fail_box_max=int(box[2])
                    fail_box_center= int((fail_box_min + fail_box_max)/2)
                    print(fail_box_center)
                    if fail_box_center <325 and fail_box_center>315 and not fail_detected:
                        errores=errores+1
                        fail_detected=True
                    else:
                        fail_detected=False
                cv2.putText(frame, '{}: {}'.format(labels[i], round(scores[i].item(), 2)), (int(box[0]), int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('Detecto', frame)

        # If the 'q' or ESC key is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyWindow('Detecto')
    video.release()

detect_live = detect_live2

@cli.command(help="Funcion que entrena y genera un modelo de deteccion de galletas")
def train_model():

    print("Entrenando modelo")

    custom_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(800),
        transforms.ColorJitter(saturation=0.3),
        transforms.ToTensor(),
        utils.normalize_transform(),
    ])

    dataset = Dataset('galletas proyecto',transform=custom_transforms)
    labels = ['Galleta', 'Galleta_rota']
    model = Model(labels)
    model.fit(dataset, verbose=True)
    val_dataset = Dataset('val_dataset')
    losses = model.fit(dataset, val_dataset, epochs=15, learning_rate=0.01,
                       gamma=0.2, lr_step_size=5, verbose=True)
    plt.plot(losses)
    plt.show()
    model.save("modelo.pth")
    print('Modelo creado y guardado')

@cli.command(help='Funcion que utiliza un modelo generado para probar con un video')
def use_model():
    print('Usando modelo para probar')
    model = Model.load('modelo.pth', ['Galleta', 'Galleta_rota'])
    print("Analizando video")
    detect_video(model, 'galletas2.mp4', 'galletas_analizadas2.avi')
    print("Final del analisis")

if __name__ == '__main__':
    cli()



