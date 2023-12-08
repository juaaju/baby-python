from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import torch

###########LANDMARK#############
# Fungsi Landmark
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

#Fungsi Menghitung Jarak
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

#Fungsi Ngambil Koordinat Pose
def get_input_point(image):
  base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_lite.task')
  options = vision.PoseLandmarkerOptions(
      base_options=base_options,
      output_segmentation_masks=True)
  detector = vision.PoseLandmarker.create_from_options(options)
  image = mp.Image.create_from_file(image)

  detection_result = detector.detect(image)
  pose_landmarks = detection_result.pose_landmarks

  nose = pose_landmarks[0][0]
  right_shoulder = pose_landmarks[0][12]
  right_elbow = pose_landmarks[0][14]
  right_wrist = pose_landmarks[0][16]
  left_shoulder = pose_landmarks[0][11]
  left_elbow = pose_landmarks[0][13]
  left_wrist = pose_landmarks[0][15]
  left_hip = pose_landmarks[0][23]
  left_knee = pose_landmarks[0][25]
  left_ankle = pose_landmarks[0][27]
  right_hip = pose_landmarks[0][24]
  right_knee = pose_landmarks[0][26]
  right_ankle = pose_landmarks[0][28]

  right_hand = calculate_distance(right_shoulder.x*image.width, right_shoulder.y*image.height, right_elbow.x*image.width, right_elbow.y*image.height) + calculate_distance(right_elbow.x*image.width, right_elbow.y*image.height, right_wrist.x*image.width, right_wrist.y*image.height)

  left_hand = calculate_distance(left_shoulder.x*image.width, left_shoulder.y*image.height, left_elbow.x*image.width, left_elbow.y*image.height) + calculate_distance(left_elbow.x*image.width, left_elbow.y*image.height, left_wrist.x*image.width, left_wrist.y*image.height)

  right_foot = calculate_distance(right_hip.x*image.width, right_hip.y*image.height, right_knee.x*image.width, right_knee.y*image.height) + calculate_distance(right_knee.x*image.width, right_knee.y*image.height, right_ankle.x*image.width, right_ankle.y*image.height)

  left_foot = calculate_distance(left_hip.x*image.width, left_hip.y*image.height, left_knee.x*image.width, left_knee.y*image.height) + calculate_distance(left_knee.x*image.width, left_knee.y*image.height, left_ankle.x*image.width, left_ankle.y*image.height)

  coords = np.array([
      [nose.x*image.width,nose.y*image.height],
      [right_shoulder.x*image.width,right_shoulder.y*image.height],
      [left_shoulder.x*image.width,left_shoulder.y*image.height],
      [right_elbow.x*image.width,right_elbow.y*image.height],
      [left_elbow.x*image.width,left_elbow.y*image.height],
      [right_wrist.x*image.width,right_wrist.y*image.height],
      [left_wrist.x*image.width,left_wrist.y*image.height],
      [right_hip.x*image.width,right_hip.y*image.height],
      [left_hip.x*image.width,left_hip.y*image.height],
      [right_knee.x*image.width,right_knee.y*image.height],
      [left_knee.x*image.width,left_knee.y*image.height],
      [right_ankle.x*image.width,right_ankle.y*image.height],
      [left_ankle.x*image.width,left_ankle.y*image.height],
      ])
  return coords, right_foot, left_foot, right_hand, left_hand

############PERSIAPAN SGEMENTATION##################
#load model
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "model/sam_vit_b_01ec64.pth"
model_type = "vit_b"



sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

predictor = SamPredictor(sam)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

################Yolo Detection Koin####################
def detect(img):
    model = torch.hub.load('.', 'custom', path='model/best2.pt', source='local', force_reload=True)
    # Image
    # Inference
    results = model(img)

    # Results, change the flowing to: results.show()
    df = results.pandas().xyxy[0]  # or .show(), .save(), .crop(), .pandas(), etc
    df = df[df['confidence'] > 0.5]
    df['x'] = df['xmax'] - df['xmin']
    df['y'] = df['ymax'] - df['ymin']
    df['x_tengah'] = (df['xmin'] + df['xmax']) / 2
    df['y_tengah'] = (df['ymin'] + df['ymax']) / 2
    df = df.sort_values('x')
    df = df.reset_index()
    x = df['x_tengah'][0]
    y = df['y_tengah'][0]
    # print(df)
    coords = np.array([[x,y]])
    lists = [df['xmin'][0], df['ymin'][0], df['xmax'][0], df['ymax'][0]]
    width = lists[2] - lists[0]
    height = lists[3] - lists[1]
    if width > height:
        width = height
    else:
        pass

    return coords, lists, width

def coef(img):
  real_coin_size = 2.7
  coin_size = detect(img)[2]
  coef = real_coin_size / coin_size
  return coef



######################MASKING#########################

def masking(input_point, input_label):
  masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

  mask_input = logits[np.argmax(scores), :, :]

  masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)
  return masks

#################Kalkulasi Parameter#####################
def tarik_garis(mask, poin_pose):
  indices = np.where(mask > 0)
  indices_x = indices[2]
  indices_y = indices[1]

    # Cari nilai indices_x yang sama dengan nilai poin_pose indeks ke-0
  val_poin_pose = int(poin_pose[0])  # Dapatkan nilai di indeks pertama dari poin_pose
  matching_indices = [index for index, value in enumerate(indices_x) if value == val_poin_pose]  # Dapatkan indeks dari nilai val_poin_pose di indices_x
  matching_indices_y_values = [indices_y[index] for index in matching_indices]  # Ambil nilai indices_y yang memiliki indeks yang sama dengan indices_x yang cocok

  #print("Nilai indices_x yang cocok dengan,", int(poin_pose[0]), "adalah", matching_indices)
  #print("Nilai indices_y dengan indeks yang sama:", matching_indices_y_values)
  #print(indices[1])
  #print(poin_pose)

  # Temukan nilai maksimum dari daftar matching_indices_y_values
  if matching_indices_y_values:
      nilai_maksimum = max(matching_indices_y_values)
      nilai_minimum = min(matching_indices_y_values)
  else:
      print("Tidak ada nilai yang cocok")

  return abs(nilai_maksimum-nilai_minimum)

def calculate_bbox_from_mask(mask):
    indices = np.where(mask > 0)
    y_min = np.min(indices[0])
    y_max = np.max(indices[0])
    x_min = np.min(indices[1])
    x_max = np.max(indices[1])
    y = y_max - y_min
    x = x_max - x_min
    if y > x:
        width = y
    else:
        width = x
    return [x_min, y_min, x_max, y_max], width

def perhitungan(sb_mayor, sb_minor):
  #hitung keliling elips
  keliling_elips=0.5 * math.pi *(sb_mayor + sb_minor)
  return keliling_elips

def perhitungan2(mask, mask2, img1, koin1, koin2):
  #hitung tinggi dada dengan masking segmentation
  #body_coord, _ = calculate_bbox_from_mask(mask2)
  #rint(body_coord)

  #y_max_body=body_coord[1]
  #y_min_body=body_coord[3]
  #tinggi_dada=abs(y_max_body-y_min_body)*coef/10 #digunakan juga untuk lingkar perut
  #print(y_max_body)
  
  #hitung panjang badan, lebar dada, lebar perut dengan pose landmark koordinat
  coords, right_foot, left_foot, right_hand, left_hand = get_input_point(img1)
  panjang_tangan = (right_hand+left_hand)/2
  x_rightshoulder = coords[1][0]
  y_rightshoulder = coords[1][1]
  x_leftshoulder = coords[2][0]
  y_leftshoulder = coords[2][1]
  x_righthip = coords[5][0]
  y_righthip = coords[5][1]
  x_lefthip = coords[6][0]
  y_lefthip = coords[6][1]

  head_coordinate , _ = calculate_bbox_from_mask(mask[0])
  #print(head_coordinate)
  x_min_kepala=head_coordinate[0]
  #print(x_min_kepala)
  x_max_kepala=head_coordinate[2]
  #print(x_max_kepala)
  panjang_kepala = abs(x_min_kepala-x_max_kepala)
  #print(panjang_kepala*koefisien/10)

  panjang_badan=(abs(x_rightshoulder-x_righthip))
  #print(panjang_badan*koefisien/10)

  panjang_kaki= ((right_foot+left_foot)/2) #di bagi 2 (panjang kaki keduanya dijumlah dan dibagi 2)
  #print(panjang_kaki*koefisien/10)

  badan_coord , _ = calculate_bbox_from_mask(mask2[0])
  y_min_badan=badan_coord[1]
  y_max_badan=badan_coord[3]
  tinggi_dada = abs(y_min_badan-y_max_badan)*koin2
  #print(y_min_badan, y_max_badan)

  total_bdn = (panjang_kepala + panjang_badan + panjang_kaki)*koin1
  #print("panjang bayi: ", total_bdn, "cm")

  lebar_dada=abs(y_rightshoulder-y_leftshoulder)*koin1
  #print("lebar dada: ", lebar_dada, "cm")

  lebar_pinggang=(abs(y_righthip-y_lefthip)-0.4*abs(y_righthip-y_lefthip))*koin1
  #print("lebar pinggang: ", lebar_pinggang, "cm")
  panjang_kaki = panjang_kaki*koin1
  panjang_tangan = panjang_tangan*koin1
  return total_bdn, lebar_dada, lebar_pinggang, tinggi_dada, panjang_kepala, panjang_kaki, panjang_tangan

##############################proses dan kalkulasi################################\
#variabel tetap
input_label_kepala = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
input_label_paha = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
input_label_lengan = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

input_label_kepala2 = np.array([1, 0, 0, 0, 0, 0, 0])
input_label_paha2 = np.array([0, 0, 0, 0, 0, 1, 0])
input_label_lengan2 = np.array([0, 0, 1, 0, 0, 0, 0])
input_label_badan = np.array([0, 1, 0, 0, 0, 0, 0])

#all parameter
def all_parameter(img1, img2):
  #proses 1
  koin1=coef(img1)
  koin2=coef(img2)
  coords, _, _, _, _=get_input_point(img1)
  input_poin1 = coords
  image_read = cv2.imread(img1)
  image = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB)
  predictor.set_image(image)

  masking_kepala=masking(input_poin1, input_label_kepala)
  masking_lengan=masking(input_poin1, input_label_lengan)
  masking_paha=masking(input_poin1, input_label_paha)

  #proses 2
  coords2, _, _, _, _=get_input_point(img2)

  input_poin2 = np.array([
      [coords2[0][0], coords2[0][1]],
      [coords2[2][0], coords2[2][1]],
      [coords2[4][0], coords2[4][1]],
      [coords2[6][0], coords2[6][1]],
      [coords2[8][0], coords2[8][1]],
      [coords2[10][0], coords2[10][1]],
      [coords2[12][0], coords2[12][1]],
      ])
  
  image_read2 = cv2.imread(img2)
  image2 = cv2.cvtColor(image_read2, cv2.COLOR_BGR2RGB)
  predictor.set_image(image2)

  masking_kepala2=masking(input_poin2, input_label_kepala2)
  masking_lengan2=masking(input_poin2, input_label_lengan2)
  masking_paha2=masking(input_poin2, input_label_paha2)
  masking_badan=masking(input_poin2, input_label_badan)

  #perhitungan
  garis_kepala_depan=tarik_garis(masking_kepala, input_poin1[0])*koin1
  garis_lengan_depan=tarik_garis(masking_lengan, input_poin1[4])*koin1
  garis_paha_depan=tarik_garis(masking_paha, input_poin1[10])*koin1
  garis_kepala_samping=tarik_garis(masking_kepala2, input_poin2[0])*koin2
  garis_lengan_samping=tarik_garis(masking_lengan2, input_poin2[2])*koin2
  garis_paha_samping=tarik_garis(masking_paha2, input_poin2[5])*koin2
  garis_badan_samping=tarik_garis(masking_badan, input_poin2[1])*koin2

  total_bdn, lebar_dada, lebar_pinggang, tinggi_dada, panjang_kepala, panjang_kaki, panjang_tangan = perhitungan2(masking_kepala, masking_badan, img1, koin1, koin2)

  lingkar_kepala = perhitungan(garis_kepala_depan, garis_kepala_samping)
  lingkar_lengan = perhitungan(garis_lengan_depan, garis_lengan_samping)
  lingkar_paha = perhitungan(garis_paha_depan, garis_paha_samping)
  lingkar_perut= perhitungan(lebar_pinggang, tinggi_dada)
  lingkar_dada= perhitungan(lebar_dada, tinggi_dada)
  total_panjang_bayi= total_bdn
  print(f'Lingkar kepala: {lingkar_kepala}, Lingkar lengan: {lingkar_lengan}, Lingkar paha: {lingkar_paha}, Lingkar perut: {lingkar_perut}, Lingkar dada: {lingkar_dada}, Total panjang bayi: {total_panjang_bayi}, Panjang kaki: {panjang_kaki}, Panjang tangan: {panjang_tangan}')

  return [lingkar_kepala, lingkar_lengan, lingkar_paha, lingkar_perut, lingkar_dada, total_panjang_bayi, panjang_kaki, panjang_tangan]
# print(coef('baby5-side.jpeg')
