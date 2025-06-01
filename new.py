import streamlit as st
import numpy as np
import requests
import datetime
import json
import cv2
import mediapipe as mp

# ---------- Streamlit Page Setup ----------
st.set_page_config(page_title="Student Face System", page_icon="🎓", layout="wide")

with st.sidebar.expander("Notes / Click here 📚", expanded=False):
    st.markdown("## Installation Guide")
    st.markdown("""
    ```bash
    pip install streamlit numpy opencv-python mediapipe requests
    ```
    No need for dlib or face_recognition anymore!
    """)

    st.markdown("## User Manual")
    st.markdown("""
    ### Register Tab
    1. Enter name + ID.
    2. Take photo with face visible.
    3. Click **Register**.

    ### Login Tab
    1. Take a photo.
    2. Click **Login**.

    ⚠️ One face only, use good lighting.
    """)

# ---------- Config ----------
GAS_ENDPOINT = "https://script.google.com/macros/s/AKfycbz85q3-5fifClgDUqGQ6hrN3cDa3AgywAwzUSf7Q7VMWz-GI56RWV0IchCpyE7Q-jJjuQ/exec"
mp_face = mp.solutions.face_mesh

# ---------- Functions ----------
@st.cache_data(show_spinner=False)
def fetch_registered():
    try:
        resp = requests.get(GAS_ENDPOINT, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        names, encs, valid_data = [], [], []
        for d in data:
            try:
                arr = np.fromstring(d["encoding"], sep=",")
                if arr.shape[0] != 936:
                    st.warning(f"⚠️ Skipped corrupted encoding for: {d['name']}")
                    continue
                names.append(d["name"])
                encs.append(arr)
                valid_data.append(d)
            except Exception:
                st.warning(f"⚠️ Failed to parse encoding for: {d.get('name', 'Unknown')}")
        return names, encs, valid_data
    except Exception as e:
        st.error(f"Failed to fetch registered users: {e}")
        st.stop()

def post_student(row):
    try:
        requests.post(GAS_ENDPOINT, json=row, timeout=10).raise_for_status()
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
        st.stop()

def extract_landmarks(image):
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image)
        if not results.multi_face_landmarks:
            return None
        h, w, _ = image.shape
        landmarks = results.multi_face_landmarks[0].landmark
        coords = [(lm.x * w, lm.y * h) for lm in landmarks]
        return np.array(coords).flatten()  # 936 values

def draw_face_boxes(image, landmarks):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for lm in landmarks:
        for x, y in lm:
            cv2.circle(img_bgr, (int(x), int(y)), 1, (0, 255, 0), -1)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def compare_landmarks(known_encs, test_enc, threshold=15.0):
    for idx, enc in enumerate(known_encs):
        dist = np.linalg.norm(enc - test_enc)
        if dist < threshold:
            return idx
    return None

# ---------- App Layout ----------
st.title("🎓 Student Face System — MediaPipe Edition")

names_known, encs_known, full_data = fetch_registered()
tab_reg, tab_log = st.tabs(["📝 Register", "✅ Login"])

# ---------- Register Tab ----------
with tab_reg:
    st.subheader("Register New Student")
    name = st.text_input("Full Name")
    sid = st.text_input("Student ID")
    img_file = st.camera_input("Take a photo for registration")

    reg_landmarks = None
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        reg_landmarks = extract_landmarks(image_rgb)

        if reg_landmarks is not None:
            st.image(draw_face_boxes(image_rgb, [reg_landmarks.reshape(-1, 2)]), caption="Detected Face")
        else:
            st.warning("⚠️ No face detected.")

    if st.button("Register", disabled=not (name.strip() and sid.strip() and reg_landmarks is not None)):
        match_idx = compare_landmarks(encs_known, reg_landmarks)
        if match_idx is not None:
            st.error(f"❌ Already registered as {names_known[match_idx]}")
            st.stop()

        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "student_id": sid.strip(),
            "name": name.strip(),
            "encoding": ",".join(map(str, reg_landmarks.tolist())),
        }
        post_student(row)
        st.success("✅ Registered & saved to Google Sheet")
        fetch_registered.clear()
        names_known, encs_known, full_data = fetch_registered()

    with st.expander("📄 View registered students"):
        st.dataframe(full_data, use_container_width=True)

# ---------- Login Tab ----------
with tab_log:
    st.subheader("Student Login / Check-in")
    img_file = st.camera_input("Take a photo for login")
    login_landmarks = None

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        login_landmarks = extract_landmarks(image_rgb)

        if login_landmarks is not None:
            st.image(draw_face_boxes(image_rgb, [login_landmarks.reshape(-1, 2)]), caption="Detected Face")
        else:
            st.warning("⚠️ No face detected.")

    if st.button("Login", disabled=login_landmarks is None):
        if not encs_known:
            st.error("❌ No students registered yet.")
            st.stop()

        match_idx = compare_landmarks(encs_known, login_landmarks)
        if match_idx is not None:
            st.success(f"✅ Welcome back, {names_known[match_idx]}!")
            st.session_state["logged_in"] = names_known[match_idx]
        else:
            st.error("❌ Face not recognised. Please register.")

    if "logged_in" in st.session_state:
        st.markdown(f"### Logged in as: {st.session_state['logged_in']}")
        if st.button("Log out"):
            del st.session_state["logged_in"]
            st.experimental_rerun()
