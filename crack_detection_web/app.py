import os, cv2, json, shutil, uuid, numpy as np, imageio_ffmpeg
from datetime import datetime
from functools import wraps
from collections import defaultdict
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-please-change-in-production')

UPLOAD_FOLDER  = 'static/uploads'
SESSIONS_ROOT  = 'static/sessions'          # mỗi lần phân tích = 1 folder session
METADATA_FILE  = 'static/metadata.json'     # danh sách session
USERS_FILE     = 'static/auth-service/users.json'        # danh sách tài khoản

# 34 tỉnh/thành phố trực thuộc Trung ương (theo Nghị quyết 202/2025/QH15, hiệu lực từ 12/6/2025)
VN_PROVINCES = [
    'An Giang', 'Bắc Ninh', 'Cà Mau', 'Cao Bằng', 'Cần Thơ',
    'Đà Nẵng', 'Đắk Lắk', 'Điện Biên', 'Đồng Nai', 'Đồng Tháp',
    'Gia Lai', 'Hà Nội', 'Hà Tĩnh', 'Hải Phòng', 'Hồ Chí Minh',
    'Huế', 'Hưng Yên', 'Khánh Hòa', 'Lai Châu', 'Lạng Sơn',
    'Lào Cai', 'Lâm Đồng', 'Nghệ An', 'Ninh Bình', 'Phú Thọ',
    'Quảng Ngãi', 'Quảng Ninh', 'Quảng Trị', 'Sơn La', 'Tây Ninh',
    'Thái Nguyên', 'Thanh Hóa', 'Tuyên Quang', 'Vĩnh Long',
]

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
for d in [UPLOAD_FOLDER, SESSIONS_ROOT, os.path.dirname(USERS_FILE)]:
    os.makedirs(d, exist_ok=True)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'weights',
    'best.pt'
)

# ── Nhãn & mức độ ────────────────────────────────────────────────────────────

# Mỗi nhãn có ngưỡng riêng (% diện tích mask / tổng frame)
LABEL_VI = {
    'nut_tuong': 'Nứt tường',
    'bong_troc': 'Bong tróc',
    'moc':       'Mốc',
}

def severity_for_label(label, pct):
    """
    Ngưỡng mức độ tùy từng loại lỗi:
      - nut_tuong: nguy hiểm kết cấu → ngưỡng thấp hơn
      - bong_troc: ảnh hưởng bề mặt → ngưỡng trung bình
      - moc:       ảnh hưởng thẩm mỹ/vệ sinh → ngưỡng cao hơn
    """
    if pct == 0:
        return 'An toàn', 'green'

    if label == 'nut_tuong':
        if pct < 0.5:  return 'Nhẹ',        'blue'
        if pct < 2.0:  return 'Trung bình',  'orange'
        return              'Nghiêm trọng', 'red'

    if label == 'bong_troc':
        if pct < 1.0:  return 'Nhẹ',        'blue'
        if pct < 4.0:  return 'Trung bình',  'orange'
        return              'Nghiêm trọng', 'red'

    # moc và nhãn khác
    if pct < 2.0:  return 'Nhẹ',        'blue'
    if pct < 8.0:  return 'Trung bình',  'orange'
    return              'Nghiêm trọng', 'red'

COLOR_ORDER = {'red': 0, 'orange': 1, 'blue': 2, 'green': 3}

def worst_color(colors):
    return min(colors, key=lambda c: COLOR_ORDER.get(c, 9))

# ── Metadata ─────────────────────────────────────────────────────────────────

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_metadata(data):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── Tài khoản người dùng ─────────────────────────────────────────────────────

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_users(data):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def find_user_by_email(email):
    email = (email or '').strip().lower()
    return next((u for u in load_users() if u['email'] == email), None)

def find_user_by_id(uid):
    return next((u for u in load_users() if u['id'] == uid), None)

def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login', next=request.path))
        return view_func(*args, **kwargs)
    return wrapped

# ── Màu vẽ mask theo nhãn (BGR) ──────────────────────────────────────────────
LABEL_COLOR_BGR = {
    'nut_tuong': (0,   80,  255),   # đỏ cam
    'bong_troc': (0,   200, 255),   # vàng
    'moc':       (50,  200,  50),   # xanh lá
}
DEFAULT_COLOR_BGR = (255, 100, 0)

def draw_mask_on_crop(crop_bgr, mask_full, x1, y1, x2, y2, label, pad=8,
                       img_w=None, img_h=None):
    """
    Crop vùng bbox từ ảnh gốc, sau đó overlay mask (fill mờ + contour) lên crop.
    mask_full: numpy array shape (H_orig, W_orig) hoặc (H_model, W_model) — bool/float.
    """
    color_bgr = LABEL_COLOR_BGR.get(label, DEFAULT_COLOR_BGR)

    # Resize mask về kích thước ảnh gốc nếu cần
    if mask_full.shape != (img_h, img_w):
        mask_resized = cv2.resize(
            mask_full.astype(np.uint8),
            (img_w, img_h),
            interpolation=cv2.INTER_NEAREST
        )
    else:
        mask_resized = mask_full.astype(np.uint8)

    # Vùng crop (có padding)
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(img_w, x2 + pad)
    cy2 = min(img_h, y2 + pad)

    crop      = crop_bgr[cy1:cy2, cx1:cx2].copy()
    mask_crop = mask_resized[cy1:cy2, cx1:cx2]

    # Fill bán trong suốt lên vùng mask
    overlay = crop.copy()
    overlay[mask_crop > 0] = color_bgr
    cv2.addWeighted(overlay, 0.35, crop, 0.65, 0, crop)

    # Vẽ contour viền rõ nét
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, color_bgr, 2)

    return crop, (cx1, cy1, cx2, cy2)


def crop_severity(mask_full, x1, y1, x2, y2, img_w, img_h, label):
    """
    Tính % diện tích mask so với diện tích bbox của detection đó.
    Đây là chỉ số "mật độ lỗi" trong vùng đó — chính xác hơn so với % toàn frame.
    """
    if mask_full is None:
        return 0.0
    if mask_full.shape != (img_h, img_w):
        m = cv2.resize(mask_full.astype(np.uint8), (img_w, img_h),
                       interpolation=cv2.INTER_NEAREST)
    else:
        m = mask_full.astype(np.uint8)

    bbox_mask = m[y1:y2, x1:x2]
    bbox_area = max((y2 - y1) * (x2 - x1), 1)
    mask_px   = int(np.sum(bbox_mask > 0))
    return round((mask_px / bbox_area) * 100, 2)


# ── Xử lý ẢNH ────────────────────────────────────────────────────────────────

def process_image(src_path, filename):
    sid      = datetime.now().strftime('%Y%m%d%H%M%S%f')
    sdir     = os.path.join(SESSIONS_ROOT, sid)
    crop_dir = os.path.join(sdir, 'crops')
    os.makedirs(crop_dir, exist_ok=True)

    results = model(src_path)[0]
    img_bgr = cv2.imread(src_path)
    H, W    = img_bgr.shape[:2]

    # Ảnh toàn cảnh annotated
    annotated_path = os.path.join(sdir, 'annotated.jpg')
    results.save(filename=annotated_path)

    boxes = results.boxes
    masks = results.masks
    names = results.names

    # label_stats lưu: tổng pixel mask (cho % toàn ảnh) + danh sách crop
    label_stats = defaultdict(lambda: {
        'total_mask_px': 0,
        'count': 0,
        'crops': [],           # mỗi phần tử: {file, conf, bbox_pct, severity, color}
    })

    for i, box in enumerate(boxes):
        cls_id       = int(box.cls[0].item())
        label        = names[cls_id]
        conf         = round(float(box.conf[0].item()), 2)
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())

        mask_np = None
        if masks is not None and i < len(masks.data):
            mask_np = masks.data[i].cpu().numpy()   # shape: (H_model, W_model), float 0‒1
            label_stats[label]['total_mask_px'] += int(
                np.sum(cv2.resize(mask_np.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST) > 0)
            )

        # % mật độ lỗi trong bbox này
        bbox_pct = crop_severity(mask_np, x1, y1, x2, y2, W, H, label)
        sev, col = severity_for_label(label, bbox_pct)

        count = label_stats[label]['count'] + 1
        label_stats[label]['count'] = count

        # Crop có mask overlay
        if mask_np is not None:
            crop_out, _ = draw_mask_on_crop(img_bgr, mask_np, x1, y1, x2, y2,
                                             label, pad=10, img_w=W, img_h=H)
        else:
            pad = 10
            crop_out = img_bgr[max(0,y1-pad):min(H,y2+pad),
                                max(0,x1-pad):min(W,x2+pad)].copy()

        cname = f"{label}_{count:03d}_conf{int(conf*100)}_pct{int(bbox_pct)}.jpg"
        cv2.imwrite(os.path.join(crop_dir, cname), crop_out)

        label_stats[label]['crops'].append({
            'file':     cname,
            'conf':     conf,
            'bbox_pct': bbox_pct,
            'severity': sev,
            'color':    col,
        })

    # Tổng hợp từng nhãn: % toàn ảnh + severity của crop nặng nhất
    label_results = {}
    for label, stat in label_stats.items():
        total_pct  = round((stat['total_mask_px'] / (W * H)) * 100, 2)
        sev_total, col_total = severity_for_label(label, total_pct)

        # Worst crop severity (màu nặng nhất trong nhãn này)
        worst = worst_color([c['color'] for c in stat['crops']]) if stat['crops'] else col_total

        label_results[label] = {
            'label_vi':   LABEL_VI.get(label, label),
            'count':      stat['count'],
            'percentage': total_pct,          # % toàn ảnh (để so sánh giữa các ảnh)
            'severity':   sev_total,
            'color':      worst,              # màu = nặng nhất trong các crop
            'crops':      stat['crops'],      # mỗi crop có severity riêng
        }

    overall_color = worst_color([v['color'] for v in label_results.values()]) \
                    if label_results else 'green'

    return {
        'id':            sid,
        'name':          os.path.splitext(filename)[0],
        'source_file':   filename,
        'is_video':      False,
        'annotated':     f'{sid}/annotated.jpg',
        'label_results': label_results,
        'overall_color': overall_color,
        'created_at':    datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'session_dir':   sid,
    }

# ── Xử lý VIDEO ──────────────────────────────────────────────────────────────

def process_video(src_path, filename):
    sid  = datetime.now().strftime('%Y%m%d%H%M%S%f')
    sdir = os.path.join(SESSIONS_ROOT, sid)
    os.makedirs(sdir, exist_ok=True)

    cap = cv2.VideoCapture(src_path)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    label_stats = defaultdict(lambda: {
        'max_frame_px': 0,
        'count':  0,
        'crops':  [],
        'seen':   set(),
    })
    annotated_frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res           = model(frame.copy())[0]
        annotated_bgr = res.plot(conf=0.25, masks=True, boxes=True)
        annotated_frames.append(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))

        names = res.names
        boxes = res.boxes
        masks = res.masks

        # Cập nhật max pixel/frame cho mỗi nhãn
        frame_label_px = defaultdict(int)
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            label  = names[cls_id]
            if masks is not None and i < len(masks.data):
                m = masks.data[i].cpu().numpy()
                m_full = cv2.resize(m.astype(np.uint8), (W, H),
                                    interpolation=cv2.INTER_NEAREST)
                frame_label_px[label] += int(np.sum(m_full > 0))
        for label, px in frame_label_px.items():
            if px > label_stats[label]['max_frame_px']:
                label_stats[label]['max_frame_px'] = px

        # Crop đại diện với mask overlay
        for i, box in enumerate(boxes):
            cls_id       = int(box.cls[0].item())
            label        = names[cls_id]
            conf         = round(float(box.conf[0].item()), 2)
            if conf < 0.35:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())

            # Chống crop trùng vị trí (grid 10×10)
            key = (round((x1+x2)/2/W, 1), round((y1+y2)/2/H, 1))
            if key in label_stats[label]['seen']:
                continue
            if len(label_stats[label]['crops']) >= 10:
                continue
            label_stats[label]['seen'].add(key)

            mask_np = None
            if masks is not None and i < len(masks.data):
                mask_np = masks.data[i].cpu().numpy()

            # % mật độ lỗi trong bbox → severity riêng cho crop này
            bbox_pct = crop_severity(mask_np, x1, y1, x2, y2, W, H, label)
            sev, col = severity_for_label(label, bbox_pct)

            crop_dir = os.path.join(sdir, 'crops', label)
            os.makedirs(crop_dir, exist_ok=True)

            if mask_np is not None:
                crop_out, _ = draw_mask_on_crop(frame, mask_np, x1, y1, x2, y2,
                                                 label, pad=10, img_w=W, img_h=H)
            else:
                pad = 10
                crop_out = frame[max(0,y1-pad):min(H,y2+pad),
                                  max(0,x1-pad):min(W,x2+pad)].copy()

            count = label_stats[label]['count'] + 1
            label_stats[label]['count'] = count
            cname = f"f{frame_idx:04d}_obj{count:02d}_conf{int(conf*100)}_pct{int(bbox_pct)}.jpg"
            cv2.imwrite(os.path.join(crop_dir, cname), crop_out)
            label_stats[label]['crops'].append({
                'file':     cname,
                'conf':     conf,
                'frame':    frame_idx,
                'bbox_pct': bbox_pct,
                'severity': sev,
                'color':    col,
            })

        frame_idx += 1
    cap.release()

    # Ghi video annotated (H.264 MP4)
    annotated_path = os.path.join(sdir, 'annotated.mp4')
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_exe
    writer = imageio_ffmpeg.write_frames(
        annotated_path, size=(W, H), fps=fps, codec='libx264',
        ffmpeg_log_level='quiet',
        output_params=['-pix_fmt','yuv420p','-movflags','+faststart','-preset','fast'],
    )
    writer.send(None)
    for f in annotated_frames:
        writer.send(f.tobytes())
    writer.close()

    # Tổng hợp từng nhãn
    label_results = {}
    for label, stat in label_stats.items():
        max_pct  = round((stat['max_frame_px'] / (W * H)) * 100, 2)
        sev, col = severity_for_label(label, max_pct)
        worst    = worst_color([c['color'] for c in stat['crops']]) if stat['crops'] else col
        label_results[label] = {
            'label_vi':   LABEL_VI.get(label, label),
            'count':      stat['count'],
            'percentage': max_pct,
            'severity':   sev,
            'color':      worst,
            'crops':      stat['crops'],
        }

    overall_color = worst_color([v['color'] for v in label_results.values()]) \
                    if label_results else 'green'

    return {
        'id':            sid,
        'name':          os.path.splitext(filename)[0],
        'source_file':   filename,
        'is_video':      True,
        'annotated':     f'{sid}/annotated.mp4',
        'label_results': label_results,
        'overall_color': overall_color,
        'created_at':    datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'session_dir':   sid,
        'total_frames':  frame_idx,
    }

# ── Routes: Xác thực người dùng ─────────────────────────────────────────────

@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('user_id'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm', '')

        if not name or not email or not password:
            flash('Vui lòng điền đầy đủ thông tin.')
            return render_template('register.html', name=name, email=email)
        if password != confirm:
            flash('Mật khẩu xác nhận không khớp.')
            return render_template('register.html', name=name, email=email)
        if len(password) < 6:
            flash('Mật khẩu phải có ít nhất 6 ký tự.')
            return render_template('register.html', name=name, email=email)
        if find_user_by_email(email):
            flash('Email này đã được đăng ký.')
            return render_template('register.html', name=name, email=email)

        users = load_users()
        user = {
            'id':            uuid.uuid4().hex,
            'name':          name,
            'email':         email,
            'password_hash': generate_password_hash(password),
            'created_at':    datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'birth_date':    '',
            'gender':        '',
            'province':      '',
            'address':       '',
            'phone':         '',
        }
        users.append(user)
        save_users(users)

        session['user_id']   = user['id']
        session['user_name'] = user['name']
        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        next_url = request.form.get('next') or url_for('index')

        user = find_user_by_email(email)
        if not user or not check_password_hash(user['password_hash'], password):
            flash('Email hoặc mật khẩu không đúng.')
            return render_template('login.html', email=email, next=next_url)

        session['user_id']   = user['id']
        session['user_name'] = user['name']
        return redirect(next_url)

    next_url = request.args.get('next') or url_for('index')
    return render_template('login.html', next=next_url)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/account')
def account():
    if not session.get('user_id'):
        return redirect(url_for('login', next=request.path))
    user = find_user_by_id(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login'))

    return render_template('account.html', user=user, provinces=VN_PROVINCES)


@app.route('/account/update', methods=['POST'])
def account_update():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    users = load_users()
    user  = next((u for u in users if u['id'] == session['user_id']), None)
    if not user:
        session.clear()
        return redirect(url_for('login'))

    name        = request.form.get('name', '').strip()
    birth_date  = request.form.get('birth_date', '').strip()
    gender      = request.form.get('gender', '').strip()
    province    = request.form.get('province', '').strip()
    address     = request.form.get('address', '').strip()
    phone       = request.form.get('phone', '').strip()

    if not name:
        flash('Họ tên không được để trống.')
        return redirect(url_for('account'))

    user['name']       = name
    user['birth_date'] = birth_date
    user['gender']     = gender
    user['province']   = province
    user['address']    = address
    user['phone']      = phone
    save_users(users)

    session['user_name'] = name
    flash('Đã cập nhật thông tin tài khoản.')
    return redirect(url_for('account'))


@app.route('/account/change-password', methods=['POST'])
def account_change_password():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    users = load_users()
    user  = next((u for u in users if u['id'] == session['user_id']), None)
    if not user:
        session.clear()
        return redirect(url_for('login'))

    current_password = request.form.get('current_password', '')
    new_password      = request.form.get('new_password', '')
    confirm_password  = request.form.get('confirm_password', '')

    if not check_password_hash(user['password_hash'], current_password):
        flash('Mật khẩu hiện tại không đúng.')
        return redirect(url_for('account'))
    if new_password != confirm_password:
        flash('Mật khẩu mới xác nhận không khớp.')
        return redirect(url_for('account'))
    if len(new_password) < 6:
        flash('Mật khẩu mới phải có ít nhất 6 ký tự.')
        return redirect(url_for('account'))

    user['password_hash'] = generate_password_hash(new_password)
    save_users(users)
    flash('Đã đổi mật khẩu thành công.')
    return redirect(url_for('account'))


@app.route('/account/delete', methods=['POST'])
def account_delete():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    password = request.form.get('password', '')
    users    = load_users()
    user     = next((u for u in users if u['id'] == session['user_id']), None)

    if not user or not check_password_hash(user['password_hash'], password):
        flash('Mật khẩu không đúng, không thể xóa tài khoản.')
        return redirect(url_for('account'))

    users = [u for u in users if u['id'] != user['id']]
    save_users(users)
    session.clear()
    return redirect(url_for('login'))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files.get('file')
        if not f or f.filename == '':
            return redirect(request.url)

        filename   = f.filename
        src_path   = os.path.join(UPLOAD_FOLDER, filename)
        f.save(src_path)
        ext = os.path.splitext(filename)[1].lower()

        if ext in ['.jpg', '.jpeg', '.png']:
            record = process_image(src_path, filename)
        elif ext in ['.mp4', '.avi', '.mov']:
            record = process_video(src_path, filename)
        else:
            return redirect(request.url)

        records = load_metadata()
        records.insert(0, record)
        save_metadata(records)

        return redirect(url_for('session_detail', session_id=record['id']))

    return render_template('index.html')


@app.route('/session/<session_id>')
def session_detail(session_id):
    records = load_metadata()
    record  = next((r for r in records if r['id'] == session_id), None)
    if not record:
        return redirect(url_for('archive'))
    return render_template('session.html', r=record)


@app.route('/archive')
def archive():
    records = load_metadata()
    return render_template('archive.html', records=records)


@app.route('/archive/rename/<sid>', methods=['POST'])
def rename_record(sid):
    new_name = request.form.get('name', '').strip()
    if not new_name:
        return jsonify({'success': False})
    records = load_metadata()
    for r in records:
        if r['id'] == sid:
            r['name'] = new_name
            break
    save_metadata(records)
    return jsonify({'success': True, 'name': new_name})


@app.route('/archive/delete/<sid>', methods=['POST'])
def delete_record(sid):
    records = load_metadata()
    target  = next((r for r in records if r['id'] == sid), None)
    if target:
        sdir = os.path.join(SESSIONS_ROOT, target['session_dir'])
        if os.path.exists(sdir):
            shutil.rmtree(sdir)
        records = [r for r in records if r['id'] != sid]
        save_metadata(records)
    return redirect(url_for('archive'))


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    """Nhận base64 JPEG từ camera, chạy YOLO, trả JSON kết quả + ảnh annotated."""
    import base64
    data = request.get_json(force=True)
    b64  = data.get('image', '').split(',')[-1]   # bỏ "data:image/jpeg;base64,"
    if not b64:
        return jsonify({'error': 'no image'}), 400

    img_bytes = base64.b64decode(b64)
    nparr     = np.frombuffer(img_bytes, np.uint8)
    frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'decode failed'}), 400

    H, W = frame.shape[:2]
    res  = model(frame)[0]

    annotated_bgr = res.plot(conf=0.25, masks=True, boxes=True)
    _, buf = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    annotated_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

    # Tổng hợp kết quả theo nhãn
    names  = res.names
    boxes  = res.boxes
    masks  = res.masks
    label_data = defaultdict(lambda: {'mask_px': 0, 'count': 0})

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        label  = names[cls_id]
        label_data[label]['count'] += 1
        if masks is not None and i < len(masks.data):
            m = masks.data[i].cpu().numpy()
            m_full = cv2.resize(m.astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST)
            label_data[label]['mask_px'] += int(np.sum(m_full > 0))

    detections = []
    for label, stat in label_data.items():
        pct = round((stat['mask_px'] / (W * H)) * 100, 2)
        sev, col = severity_for_label(label, pct)
        detections.append({
            'label':    label,
            'label_vi': LABEL_VI.get(label, label),
            'count':    stat['count'],
            'pct':      pct,
            'severity': sev,
            'color':    col,
        })

    overall = worst_color([d['color'] for d in detections]) if detections else 'green'

    return jsonify({
        'annotated': annotated_b64,
        'detections': detections,
        'overall_color': overall,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
