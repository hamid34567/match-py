import cv2
import face_recognition
import pickle
import numpy as np
from flask import Flask, Response, render_template, send_file
import os
import threading
import time
import sys
import queue 
import io 

# Mengatur encoding output konsol ke UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Inisialisasi aplikasi Flask
app = Flask(_name_)

# Variabel global untuk menyimpan objek webcam dan statusnya
video_stream = None
video_stream_lock = threading.Lock()

# --- Queue untuk berbagi data perbandingan ---
# generate_frames akan memasukkan data ke queue ini
# compare_frame akan mengambil data dari queue ini
comparison_data_queue = queue.Queue(maxsize=1) # Maxsize 1 agar hanya menyimpan data terbaru

# --- Variabel global untuk menyimpan data perbandingan terakhir yang berhasil ---
# Ini akan membuat gambar perbandingan "lengket"
last_successful_comparison_data = (None, None, "Tidak Ada", 0.0)
last_detection_time = time.time() # Waktu terakhir deteksi wajah yang berhasil

# --- Bagian 1: Memuat Enkripsi Waifu ---
waifu_encodings = []
waifu_names = []
try:
    with open("waifu_encodings.pickle", "rb") as f:
        waifu_encodings, waifu_names = pickle.load(f)
    print("File 'waifu_encodings.pickle' berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'waifu_encodings.pickle' tidak ditemukan.")
    print("Pastikan Anda telah membuat file ini dengan enkripsi wajah waifu yang benar.")
    print("Program akan berhenti.")
    exit()
except Exception as e:
    print(f"Error saat memuat file pickle: {e}")
    exit()

# --- Bagian 2: Fungsi Pembantu ---
def face_distance_to_confidence(face_distance, threshold=0.6):
    if face_distance > threshold:
        return round(100 * (1.0 - face_distance), 2)
    else:
        return round(100 * (1.0 - face_distance * face_distance), 2)

def get_waifu_image_path(waifu_name, dataset_base_path="waifu_dataset"):
    waifu_folder = os.path.join(dataset_base_path, waifu_name)
    if os.path.isdir(waifu_folder):
        for filename in os.listdir(waifu_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                found_path = os.path.join(waifu_folder, filename)
                return found_path
    return None

def initialize_webcam():
    global video_stream
    with video_stream_lock:
        if video_stream is None or not video_stream.isOpened():
            print("[INFO] Mencoba membuka webcam (indeks 1) di startup aplikasi...")
            temp_cap = cv2.VideoCapture(0)
            
            if not temp_cap.isOpened():
                print("[INFO] Webcam indeks 1 gagal. Mencoba membuka webcam (indeks 0) di startup aplikasi...")
                temp_cap = cv2.VideoCapture(0)
            
            if not temp_cap.isOpened():
                print("Error: Tidak dapat membuka webcam saat startup. Pastikan webcam terhubung dengan benar, drivernya terinstal, dan tidak sedang digunakan oleh aplikasi lain (misalnya Zoom, Skype, aplikasi Kamera).")
                temp_cap.release()
                video_stream = None
            else:
                video_stream = temp_cap
                video_stream.set(cv2.CAP_PROP_FPS, 30)
                video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print("Webcam berhasil diinisialisasi saat startup.")
        else:
            print("Webcam sudah aktif dari inisialisasi sebelumnya.")

# --- Bagian 3: Fungsi Generator untuk Streaming Video Utama ---
def generate_frames():
    global video_stream, last_successful_comparison_data, last_detection_time

    with video_stream_lock:
        if video_stream is None or not video_stream.isOpened():
            print("Webcam tidak tersedia. Streaming tidak dapat dimulai.")
            return

    active_face_trackers = []
    frame_count = 0
    DETECTION_INTERVAL = 5

    while True:
        try:
            with video_stream_lock:
                if video_stream is None or not video_stream.isOpened():
                    print("Webcam tidak lagi tersedia atau ditutup. Menghentikan streaming.")
                    break

                ret, frame = video_stream.read()

            if not ret:
                print("Gagal membaca frame dari webcam. Streaming berhenti.")
                break

            if frame is None or frame.size == 0:
                print("Peringatan: Frame kosong atau tidak valid diterima dari webcam. Melewatkan frame ini.")
                continue

            current_user_face_img_rgb = None
            current_matched_waifu_img_rgb = None
            current_matched_waifu_name = "Tidak Ada"
            current_matched_similarity = 0.0
            
            face_detected_in_this_frame = False

            if frame_count % DETECTION_INTERVAL == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                if small_frame is None or small_frame.size == 0:
                    print("Peringatan: small_frame kosong atau tidak valid setelah resize. Melewatkan frame ini.")
                    continue
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                if rgb_small is None or rgb_small.size == 0:
                    print("Peringatan: rgb_small kosong atau tidak valid setelah konversi warna. Melewatkan frame ini.")
                    continue

                try:
                    locations = face_recognition.face_locations(rgb_small)
                    encodings = face_recognition.face_encodings(rgb_small, locations)

                    active_face_trackers = []

                    for i, face_encoding in enumerate(encodings):
                        distances = face_recognition.face_distance(waifu_encodings, face_encoding)
                        best_match_idx = np.argmin(distances)

                        name = "Unknown"
                        similarity = 0.0

                        if distances[best_match_idx] < 0.6:
                            name = waifu_names[best_match_idx]
                            similarity = face_distance_to_confidence(distances[best_match_idx])

                            if name != "Unknown":
                                face_detected_in_this_frame = True # Tandai bahwa wajah terdeteksi
                                top, right, bottom, left = locations[i]
                                top *= 4
                                right *= 4
                                bottom *= 4
                                left *= 4

                                user_face_cropped = frame[top:bottom, left:right]
                                if user_face_cropped.size > 0:
                                    # --- Memperbesar ukuran gambar wajah yang dipotong ---
                                    user_face_resized = cv2.resize(user_face_cropped, (350, 350), interpolation=cv2.INTER_AREA) 
                                    current_user_face_img_rgb = cv2.cvtColor(user_face_resized, cv2.COLOR_BGR2RGB)

                                    waifu_img_path = get_waifu_image_path(name)
                                    if waifu_img_path:
                                        waifu_img_bgr = cv2.imread(waifu_img_path)
                                        if waifu_img_bgr is not None and waifu_img_bgr.size > 0:
                                            if waifu_img_bgr.dtype != np.uint8:
                                                waifu_img_bgr = waifu_img_bgr.astype(np.uint8)
                                            # --- Memperbesar ukuran gambar waifu yang dipotong ---
                                            waifu_img_resized = cv2.resize(waifu_img_bgr, (350, 350), interpolation=cv2.INTER_AREA) 
                                            current_matched_waifu_img_rgb = cv2.cvtColor(waifu_img_resized, cv2.COLOR_BGR2RGB)
                                        else:
                                            print(f"DEBUG: cv2.imread gagal memuat gambar waifu: {waifu_img_path}")
                                    else:
                                        print(f"DEBUG: Path gambar waifu tidak ditemukan untuk {name}.")
                                    
                                    current_matched_waifu_name = name
                                    current_matched_similarity = similarity
                                    
                                    # print(f"DEBUG: Data perbandingan diperbarui. User face: {current_user_face_img_rgb is not None}, Waifu face: {current_matched_waifu_img_rgb is not None}, Nama: {current_matched_waifu_name}, Kemiripan: {current_matched_similarity:.2f}%")
                                else:
                                    print(f"Peringatan: Wajah pengguna yang dipotong kosong untuk {name}.")

                        top, right, bottom, left = locations[i]
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        tracker = cv2.TrackerCSRT_create()
                        bbox = (left, top, right - left, bottom - top)

                        try:
                            tracker.init(frame, bbox)
                            active_face_trackers.append((tracker, name, similarity))
                        except Exception as e_init:
                            print(f"Error saat inisialisasi tracker: {e_init}")

                except Exception as e:
                    print(f"Error saat memproses face_recognition (locations/encodings): {e}")
                    active_face_trackers = []
                    pass

            else:
                updated_trackers = []
                for tracker, name, similarity in active_face_trackers:
                    success, bbox = tracker.update(frame)

                    if success:
                        updated_trackers.append((tracker, name, similarity))
                        face_detected_in_this_frame = True # Tandai bahwa wajah terdeteksi (melalui tracking)

                        left, top, w, h = [int(v) for v in bbox]
                        right = left + w
                        bottom = top + h

                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        
                        # --- Bagian yang menampilkan nama dan persentase ---
                        label = f"{name} ({similarity:.2f}%)"
                        if frame_count % DETECTION_INTERVAL != 0:
                            label += " (Tracked)"

                        # Dapatkan ukuran teks untuk menggambar latar belakang
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        
                        # Gambar persegi panjang terisi sebagai latar belakang teks
                        # Posisikan persegi panjang sedikit di atas kotak pembatas
                        cv2.rectangle(frame, (left, top - text_height - baseline - 10), (left + text_width, top - 10), (0, 0, 0), cv2.FILLED) # Latar belakang hitam
                        
                        # Tampilkan teks pada frame
                        cv2.putText(frame, label, (left, top - baseline - 5), # Sesuaikan posisi Y sedikit
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # Teks putih
                        # --- Akhir bagian tampilan teks ---
                    else:
                        pass
                active_face_trackers = updated_trackers

            # --- Masukkan data perbandingan ke queue ---
            # Data ini akan digunakan oleh rute /compare_frame
            if current_user_face_img_rgb is not None and current_matched_waifu_img_rgb is not None:
                try:
                    # Hapus item lama jika ada (agar selalu menyimpan data terbaru)
                    if not comparison_data_queue.empty():
                        comparison_data_queue.get_nowait()
                    comparison_data_queue.put_nowait((current_user_face_img_rgb, 
                                                    current_matched_waifu_img_rgb, 
                                                    current_matched_waifu_name, 
                                                    current_matched_similarity))
                    last_successful_comparison_data = (current_user_face_img_rgb, current_matched_waifu_img_rgb, current_matched_waifu_name, current_matched_similarity)
                    last_detection_time = time.time() # Update waktu deteksi terakhir
                    # print("DEBUG: Data perbandingan dimasukkan ke queue.") # Matikan agar tidak terlalu banyak log
                except queue.Full:
                    # Queue penuh, berarti item sebelumnya belum diambil, lewati saja
                    pass
                except Exception as e_queue:
                    print(f"Error saat memasukkan data ke queue: {e_queue}")
            else:
                # Jika tidak ada wajah yang dikenali atau gambar waifu tidak dimuat,
                # masukkan None ke queue, tetapi JANGAN reset last_successful_comparison_data
                try:
                    if not comparison_data_queue.empty():
                        comparison_data_queue.get_nowait()
                    comparison_data_queue.put_nowait((None, None, "Tidak Ada", 0.0))
                except queue.Full:
                    pass # Biarkan saja jika queue penuh dengan data lama


            frame_count += 1

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Gagal meng-encode frame ke JPEG.")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        except Exception as e:
            print(f"Error tak terduga saat memproses frame: {e}")
            break

# --- Bagian 4: Rute Flask ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    response = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/compare_frame') # Rute untuk mendapatkan satu frame perbandingan
def compare_frame():
    global last_successful_comparison_data, last_detection_time
    
    user_face_to_display = None
    waifu_face_to_display = None
    waifu_name_to_display = "Tidak Ada"
    similarity_to_display = 0.0

    try:
        # Coba ambil data terbaru dari queue (non-blocking)
        # Jika ada data baru, gunakan itu dan update last_successful_comparison_data
        temp_user_face, temp_waifu_face, temp_waifu_name, temp_similarity = comparison_data_queue.get_nowait()
        
        if temp_user_face is not None and temp_waifu_face is not None:
            user_face_to_display = temp_user_face
            waifu_face_to_display = temp_waifu_face
            waifu_name_to_display = temp_waifu_name
            similarity_to_display = temp_similarity
            last_successful_comparison_data = (user_face_to_display, waifu_face_to_display, waifu_name_to_display, similarity_to_display)
            last_detection_time = time.time() # Update waktu deteksi terakhir yang berhasil
            # print(f"DEBUG [compare_frame]: Data baru diambil dari queue. Nama: {waifu_name_to_display}")
        else:
            # Jika data dari queue tidak lengkap (misalnya None), gunakan data terakhir yang berhasil
            user_face_to_display, waifu_face_to_display, waifu_name_to_display, similarity_to_display = last_successful_comparison_data
            # print(f"DEBUG [compare_frame]: Queue kosong/data tidak lengkap, menggunakan data terakhir yang berhasil. Nama: {waifu_name_to_display}")

    except queue.Empty:
        # Jika queue kosong, gunakan data terakhir yang berhasil
        user_face_to_display, waifu_face_to_display, waifu_name_to_display, similarity_to_display = last_successful_comparison_data
        # print(f"DEBUG [compare_frame]: Queue kosong, menggunakan data terakhir yang berhasil. Nama: {waifu_name_to_display}")
    
    # Kriteria untuk menampilkan placeholder:
    # 1. Belum pernah ada deteksi yang berhasil (last_successful_comparison_data masih None)
    # ATAU
    # 2. Sudah lama tidak ada deteksi wajah yang berhasil (misal > 5 detik)
    if (user_face_to_display is None or waifu_face_to_display is None) or \
    (time.time() - last_detection_time > 5 and waifu_name_to_display == "Tidak Ada"): # 5 detik tanpa deteksi
        
        # print("DEBUG [compare_frame]: Mengirim placeholder karena tidak ada data yang valid atau sudah lama tidak ada deteksi.")
        placeholder_img = np.zeros((350, 700, 3), dtype=np.uint8) # Diperbesar placeholder
        cv2.putText(placeholder_img, "Menunggu Deteksi Wajah...", (70, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2) # Teks diperbesar
        ret, buffer = cv2.imencode('.jpg', placeholder_img)
        frame_bytes = buffer.tobytes()
        return Response(frame_bytes, mimetype='image/jpeg', headers={'Cache-Control': 'no-cache, no-store', 'must-revalidate': 'no-cache, no-store', 'Pragma': 'no-cache', 'Expires': '0'})
    
    # Jika ada data yang valid (baik baru dari queue atau yang terakhir berhasil)
    else:
        h1, w1, _ = user_face_to_display.shape
        h2, w2, _ = waifu_face_to_display.shape
        
        padding = 50 # Padding lebih besar
        # --- PERBAIKAN: Meningkatkan text_area_height untuk ruang teks yang lebih besar ---
        text_area_height = 200 # Ditingkatkan lagi dari 180
        combined_width = w1 + w2 + padding * 3
        combined_height = max(h1, h2) + padding * 2 + text_area_height # Menggunakan text_area_height baru
        
        combined_img = np.full((combined_height, combined_width, 3), 50, dtype=np.uint8)

        try:
            combined_img[padding:padding+h1, padding:padding+w1] = user_face_to_display
            combined_img[padding:padding+h2, padding + w1 + padding:padding + w1 + padding + w2] = waifu_face_to_display
        except ValueError as e:
            print(f"ERROR: ValueError saat menempatkan gambar ke combined_img: {e}")
            print(f"  combined_img shape: {combined_img.shape}")
            print(f"  user_face_to_display shape: {user_face_to_display.shape}")
            print(f"  waifu_face_to_display shape: {waifu_face_to_display.shape}")
            # Fallback to placeholder on error
            placeholder_img = np.zeros((350, 700, 3), dtype=np.uint8)
            cv2.putText(placeholder_img, "Image Combine Error!", (70, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder_img)
            frame_bytes = buffer.tobytes()
            return Response(frame_bytes, mimetype='image/jpeg', headers={'Cache-Control': 'no-cache, no-store', 'must-revalidate': 'no-cache, no-store', 'Pragma': 'no-cache', 'Expires': '0'})


        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0 # Skala font lebih besar
        font_thickness = 2
        text_color = (255, 255, 255)

        # --- PERBAIKAN: Menyesuaikan posisi Y teks lebih lanjut ---
        cv2.putText(combined_img, "Wajah Anda", (padding, padding + h1 + 90), font, font_scale, text_color, font_thickness) # Y disesuaikan
        
        waifu_text = f"Waifu: {waifu_name_to_display}"
        similarity_text = f"Kemiripan: {similarity_to_display:.2f}%"
        
        cv2.putText(combined_img, waifu_text, (padding + w1 + padding, padding + h2 + 90), font, font_scale, text_color, font_thickness) # Y disesuaikan
        cv2.putText(combined_img, similarity_text, (padding + w1 + padding, padding + h2 + 170), font, font_scale, text_color, font_thickness) # Posisi Y disesuaikan lagi (dari 150 menjadi 170)
        # --- AKHIR PERBAIKAN POSISI TEKS ---

        combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
        
        # --- DEBUG: Simpan gambar gabungan ke file sebelum encode ---
        debug_filename = "debug_combined_img.jpg"
        try:
            cv2.imwrite(debug_filename, combined_img_bgr)
            # print(f"DEBUG [compare_frame]: Menyimpan gambar gabungan ke {debug_filename}") # Matikan agar tidak terlalu banyak log
        except Exception as e_imwrite:
            print(f"DEBUG [compare_frame]: Gagal menyimpan debug_combined_img.jpg: {e_imwrite}")
        # --- END DEBUG ---

        ret, buffer = cv2.imencode('.jpg', combined_img_bgr)
        
        if not ret or len(buffer.tobytes()) == 0:
            print("DEBUG [compare_frame]: Gagal meng-encode gambar perbandingan ke JPEG (ret=False atau buffer kosong).")
            placeholder_img = np.zeros((350, 700, 3), dtype=np.uint8)
            cv2.putText(placeholder_img, "Encoding Failed!", (70, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder_img)
            frame_bytes = buffer.tobytes()
            return Response(frame_bytes, mimetype='image/jpeg', headers={'Cache-Control': 'no-cache, no-store', 'must-revalidate': 'no-cache, no-store', 'Pragma': 'no-cache', 'Expires': '0'})
        
        frame_bytes = buffer.tobytes()
        return Response(frame_bytes, mimetype='image/jpeg', headers={'Cache-Control': 'no-cache, no-store', 'must-revalidate': 'no-cache, no-store', 'Pragma': 'no-cache', 'Expires': '0'})


# --- Bagian 5: Fungsi Cleanup saat aplikasi ditutup ---
def teardown_webcam(exception=None):
    global video_stream
    with video_stream_lock:
        if video_stream is not None:
            video_stream.release()
            print("Webcam berhasil dilepaskan.")
            video_stream = None

# --- Bagian 6: Menjalankan Aplikasi Flask ---
if _name_ == '_main_':
    if not os.path.exists('templates'):
        os.makedirs('templates')

    html_file_path = os.path.join('templates', 'index.html')
    # Pastikan file 'templates/index.html' sudah ada dan berisi konten yang benar.

    initialize_webcam()

    print("\n[INFO] Menjalankan aplikasi Flask...")
    print("Akses aplikasi di: http://127.0.0.1:5000/")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        teardown_webcam()    
