import face_recognition as fr
import cv2
import os
import time
import webbrowser

# Carregar imagens de referência
known_face_encodings = []
known_face_names = []

# Adicione imagens na pasta 'images' com nomes correspondentes
image_folder = "images/"
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        print("Carregando imagem:", filename)
        img_path = os.path.join(image_folder, filename)
        image = fr.load_image_file(img_path)
        encoding = fr.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])

url = 'http://192.168.100.18:8080/video' 
video_capture = cv2.VideoCapture(url)

if not video_capture.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

print("Iniciando reconhecimento facial ao vivo...")

# Reduzir a resolução para processamento
resize_scale = 0.5

# Configurar janela de exibição
cv2.namedWindow("Reconhecimento Facial", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Reconhecimento Facial", 800, 580) 

# Controlar processamento de frames
frame_processing_interval = 10
frame_count = 0

# Controle de tempo para limitar o FPS
fps_limit = 30
prev_time = 0

# Dicionário para controle de mensagens enviadas
sent_messages = {}

while True:
    ret, frame = video_capture.read()

    if not ret or frame is None:
        print("Erro: Não foi possível capturar o frame.")
        break

    # Limitar o FPS
    current_time = time.time()
    if (current_time - prev_time) < (1.0 / fps_limit):
        continue
    prev_time = current_time

    # Processar apenas 1 a cada N frames
    frame_count += 1
    if frame_count % frame_processing_interval != 0:
        continue

    # Reduzir o tamanho do frame para processamento
    frame_small = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)

    # Converter frame reduzido para RGB
    rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    # Detectar rostos no frame reduzido
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"

        # Comparar com rostos conhecidos
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            print("Reconhecido como:", name)

            # Verificar se já enviamos mensagem para este rosto
            if name not in sent_messages or time.time() - sent_messages[name] > 60:  # 60 segundos de intervalo
                # URL de envio para o WhatsApp
                phone_number = "+558582059273"  # Substitua pelo número do destinatário
                message = f"Olá, {name} foi reconhecido pela câmera!"
                whatsapp_url = f"https://api.whatsapp.com/send?phone={phone_number}&text={message}"

                # Abrir a URL no navegador
                print(f"Enviando mensagem para {name}: {whatsapp_url}")
                webbrowser.open(whatsapp_url)

                # Atualizar o dicionário para registrar o envio
                sent_messages[name] = time.time()

        # Ajustar coordenadas para o frame original (tamanho real)
        top = int(top / resize_scale)
        right = int(right / resize_scale)
        bottom = int(bottom / resize_scale)
        left = int(left / resize_scale)

        # Desenhar um retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Reduzir o tamanho do frame exibido para melhorar a exibição
    display_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)  # Redimensionar para exibição

    # Exibir o frame na janela
    cv2.imshow("Reconhecimento Facial", display_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar as janelas
video_capture.release()
cv2.destroyAllWindows()
