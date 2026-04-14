import streamlit as st
import dashscope
from dashscope import MultiModalConversation
from PIL import Image
import base64, io, cv2, numpy as np, json, os, re
from ultralytics import YOLO

# ====================== 1. 核心配置 ======================
# 建议在实际部署时通过 st.secrets 管理 Key
dashscope.api_key = st.secrets["DASHSCOPE_API_KEY"]
YOLO_MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 80  # 置信度阈值
MAX_REASONING_ROUNDS = 3   # 最大推理/提问轮次，防止无限循环

# ====================== 2. 页面配置 ======================
st.set_page_config(page_title="面包板接线小医", layout="centered", page_icon="🔧")

# ====================== 3. 状态管理 ======================
if 'step' not in st.session_state: st.session_state.step = 'idle'
if 'wire_data' not in st.session_state: st.session_state.wire_data = None
if 'img_b64' not in st.session_state: st.session_state.img_b64 = None
if 'qa_history' not in st.session_state: st.session_state.qa_history = []
if 'sidebar_log' not in st.session_state: st.session_state.sidebar_log = []
if 'current_questions' not in st.session_state: st.session_state.current_questions = []
if 'processed_img' not in st.session_state: st.session_state.processed_img = None
if 'ai_response' not in st.session_state: st.session_state.ai_response = None
if 'confidence' not in st.session_state: st.session_state.confidence = 0
if 'current_round' not in st.session_state: st.session_state.current_round = 0

# ====================== 4. 核心功能函数 ======================
@st.cache_resource
def load_yolo_model():
    try:
        if not os.path.exists(YOLO_MODEL_PATH):
            st.error(f"模型文件不存在：{YOLO_MODEL_PATH}")
            return None
        return YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        st.error(f"模型加载失败：{e}")
        return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmax(diff)]
    rect[3] = pts[np.argmin(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def preprocess_image(uploaded_file, model):
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    output_img = cv_img.copy()
    h, w = cv_img.shape[:2]
    kernel = np.ones((3, 3), np.uint8)
    correction_success = False

    if model is not None:
        try:
            results = model(cv_img, conf=0.5, save=False, verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) >= 4:
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                corners_dict = {}
                for i in range(len(xyxy)):
                    cls = int(cls_ids[i])
                    if cls < 4:
                        x1, y1, x2, y2 = xyxy[i]
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        if cls not in corners_dict or confs[i] > corners_dict[cls][1]:
                            corners_dict[cls] = ([cx, cy], confs[i])
                if len(corners_dict) == 4:
                    pts = np.array([corners_dict[i][0] for i in range(4)], dtype="float32")
                    cv_img = four_point_transform(cv_img, pts)
                    output_img = cv_img.copy()
                    correction_success = True
        except: pass

    if not correction_success:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            if angle < -45: angle += 90
            elif angle > 45: angle -= 90
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            cv_img = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            output_img = cv_img.copy()

    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    wire_info = []
    color_ranges = {"red": [(0, 100, 100), (10, 255, 255)], "blue": [(90, 100, 100), (130, 255, 255)], "black": [(0, 0, 0), (180, 255, 50)]}
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if 100 < area < 50000:
                x, y, w_wire, h_wire = cv2.boundingRect(c)
                points = np.squeeze(c)
                if len(points) > 0:
                    p1 = tuple(points[np.argmin(np.sum(points, axis=1))])
                    p2 = tuple(points[np.argmax(np.sum(points, axis=1))])
                    status = "双端连接" if abs(p1[1] - p2[1]) < 50 else "单端悬空"
                    wire_info.append({"color": color, "p1": p1, "p2": p2, "status": status})
                    cv2.rectangle(output_img, (x, y), (x+w_wire, y+h_wire), (0, 255, 0), 2)
                    cv2.circle(output_img, p1, 4, (0, 0, 255), -1)
                    cv2.circle(output_img, p2, 4, (255, 0, 0), -1)

    output_img = cv2.convertScaleAbs(output_img, alpha=1.2, beta=10)
    pil_img = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    return pil_img, json.dumps(wire_info, ensure_ascii=False, default=str)

def extract_json(text):
    """增强版 JSON 提取，兼容 Markdown 代码块与字段缺失"""
    if not text or not text.strip():
        return {"status": "final", "confidence": 0, "sidebar_summary": "⚠️ 模型返回为空", "questions": [], "report": "AI 未生成任何文字反馈，请检查网络或重试。"}
    
    clean_text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
    match = re.search(r'\{.*\}', clean_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            data.setdefault("status", "final")
            data.setdefault("confidence", 0)
            data.setdefault("sidebar_summary", "解析成功")
            data.setdefault("questions", [])
            if not data.get("report"):
                data["report"] = "模型未提供详细报告，请查看原始分析。"
            return data
        except json.JSONDecodeError:
            pass
            
    return {"status": "final", "confidence": 0, "sidebar_summary": "⚠️ JSON 格式异常", "questions": [], "report": f"解析失败，原始文本：\n{clean_text}"}

def call_qwen_diagnosis(img_b64, wire_data, qa_history, round_num):
    qa_text = "\n".join([f"第 {i+1} 轮确认 - {q['text']} => {q['answer']}" for i, q in enumerate(qa_history)])
    
    # 🔑 核心 Prompt：加入置信度评分与多轮推理逻辑
    prompt = f"""你是面包板电路辅助诊断专家。当前是第 {round_num} 轮推理。
【已知 CV 数据】{wire_data}
【历史确认记录】{qa_text if qa_history else '无'}

【输出规则】必须且仅输出严格 JSON，禁止任何 Markdown 或额外文本：
{{"status":"question"/"final", "confidence":0-100, "sidebar_summary":"3-4 个短句带图标", "questions":[{{"id":1, "text":"...", "type":"choice"/"text", "options":["A","B"]}}], "report":"Markdown 完整报告"}}

【工作流与判断逻辑】
1. 置信度评估 (confidence)：基于视觉清晰度、遮挡情况、CV 识别成功率评估。若存在关键引脚被遮挡或反光严重，confidence 必须低于 80。
2. 提问策略 (status="question")：
   - 仅当 confidence < 80 且当前轮次 < 3 时触发。
   - 问题类型 type 可为 "choice" (需附 options 数组) 或 "text" (开放文本)。
   - 必须询问：①电路功能目标？②供电电压？③遮挡/模糊区域的具体内容。
3. 最终报告 (status="final")：
   - 当 confidence >= 80 或达到最大轮次时输出。
   - 报告必须包含：✅ 已确认事实、⚠️ 存疑区域（注明置信度低原因）、🛠️ 验证建议（如万用表测试点）。
   
【⚠️ 格式要求】
- report 字段内的换行必须用 \\n 表示，双引号必须用 \\" 转义。
- 绝不要输出 ```json 标记。"""
    
    messages = [{"role": "user", "content": [{"image": f"data:image/jpeg;base64,{img_b64}"}, {"text": prompt}]}]
    try:
        response = MultiModalConversation.call(model="qwen-vl-max", messages=messages)
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            text = content[0]["text"] if isinstance(content, list) else str(content)
            return extract_json(text)
        return {"status": "final", "confidence": 0, "sidebar_summary": "⚠️ API 调用失败", "questions": [], "report": f"API 错误：{response.message}"}
    except Exception as e:
        return {"status": "final", "confidence": 0, "sidebar_summary": "❌ 系统异常", "questions": [], "report": f"错误：{str(e)}"}

# ====================== 5. Streamlit 界面布局 ======================
st.title("🔧 面包板接线小医")

uploaded_file = st.file_uploader("📤 上传实验板照片", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    if st.session_state.step == 'idle':
        st.session_state.step = 'ready'

if st.session_state.step in ['ready', 'questioning', 'final']:
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.image(Image.open(uploaded_file), caption="📷 原始照片", use_container_width=True)
        
    with col2:
        if st.session_state.processed_img is not None:
            st.image(st.session_state.processed_img, caption="🎯 透视矫正与 CV 标注视图", use_container_width=True)
        else:
            st.info("⏳ 等待诊断启动...")
            
    with col3:
        # --- 状态：准备就绪 ---
        if st.session_state.step == 'ready':
            st.subheader("🚀 准备就绪")
            st.write("图片已加载。系统将通过 **多轮置信度评估** 确保诊断结果准确可靠。")
            if st.button("启动 AI 协同诊断", type="primary", use_container_width=True):
                with st.spinner("🧠 视觉预处理 → 导线拓扑提取 → 大模型事实清点中..."):
                    pil_img, wire_json = preprocess_image(uploaded_file, load_yolo_model())
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="JPEG")
                    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    st.session_state.processed_img = pil_img
                    st.session_state.wire_data = wire_json
                    st.session_state.img_b64 = img_b64
                    st.session_state.qa_history = []
                    st.session_state.current_round = 1
                    
                    ai_res = call_qwen_diagnosis(img_b64, wire_json, [], 1)
                    st.session_state.ai_response = ai_res
                    st.session_state.confidence = ai_res.get("confidence", 0)
                    st.session_state.sidebar_log = [f"第 1 轮：{ai_res['sidebar_summary']} (置信度 {st.session_state.confidence}%)"]
                    
                    # 判断下一步状态
                    if ai_res["status"] == "question" and st.session_state.confidence < CONFIDENCE_THRESHOLD and st.session_state.current_round < MAX_REASONING_ROUNDS:
                        st.session_state.step = 'questioning'
                        st.session_state.current_questions = ai_res.get("questions", [])
                    else:
                        st.session_state.step = 'final'
                st.rerun()

        # --- 状态：需用户确认（置信度不足） ---
        elif st.session_state.step == 'questioning':
            st.subheader(f"💡 需您协助确认 (第 {st.session_state.current_round} 轮)")
            st.info(f"当前 AI 诊断置信度为 **{st.session_state.confidence}%** (低于 {CONFIDENCE_THRESHOLD}%)。请补充以下信息以提高准确率：")
            
            with st.form(f"qa_form_round_{st.session_state.current_round}", border=False):
                answers = {}
                for q in st.session_state.current_questions:
                    q_type = q.get("type", "text")
                    if q_type == "choice" and "options" in q:
                        answers[q["id"]] = st.radio(q["text"], q["options"], index=0, key=f"q_{q['id']}")
                    else:
                        answers[q["id"]] = st.text_input(q["text"], placeholder="请输入您的回答...", key=f"q_{q['id']}")
                
                submitted = st.form_submit_button("📤 提交反馈，继续推理", type="primary")
                if submitted:
                    # 记录用户回答
                    for q in st.session_state.current_questions:
                        ans = answers[q["id"]]
                        st.session_state.qa_history.append({"text": q["text"], "answer": ans})
                    
                    # 准备下一轮
                    st.session_state.current_round += 1
                    
                    with st.spinner(f"🔄 结合您的反馈进行第 {st.session_state.current_round} 轮推理..."):
                        ai_res = call_qwen_diagnosis(st.session_state.img_b64, st.session_state.wire_data, st.session_state.qa_history, st.session_state.current_round)
                        st.session_state.ai_response = ai_res
                        st.session_state.confidence = ai_res.get("confidence", 0)
                        log_entry = f"第 {st.session_state.current_round} 轮：{ai_res['sidebar_summary']} (置信度 {st.session_state.confidence}%)"
                        st.session_state.sidebar_log.append(log_entry)
                        
                        # 再次判断状态
                        if ai_res["status"] == "question" and st.session_state.confidence < CONFIDENCE_THRESHOLD and st.session_state.current_round < MAX_REASONING_ROUNDS:
                            st.session_state.step = 'questioning'
                            st.session_state.current_questions = ai_res.get("questions", [])
                        else:
                            st.session_state.step = 'final'
                    st.rerun()

        # --- 状态：最终报告 ---
        elif st.session_state.step == 'final':
            conf = st.session_state.confidence
            st.subheader(f"📋 最终诊断报告 (置信度：{conf}%)")
            
            # 如果置信度依然很低，给出额外提示
            if conf < CONFIDENCE_THRESHOLD:
                st.warning(f"⚠️ 经过 {st.session_state.current_round} 轮推理，置信度仍为 {conf}%。报告内容基于有限视觉信息生成，建议结合万用表进行实物验证。")
            
            report_text = st.session_state.ai_response.get("report", "")
            if not report_text or report_text.strip() == "":
                st.error("⚠️ AI 未返回有效诊断文字，请尝试重新上传清晰照片。")
            else:
                st.markdown(report_text, unsafe_allow_html=True)
                
            st.divider()
            
            col_reset, col_export = st.columns([1, 1])
            with col_reset:
                if st.button("🔄 重新开始新诊断", type="secondary"):
                    st.session_state.step = 'idle'
                    st.session_state.qa_history = []
                    st.session_state.sidebar_log = []
                    st.session_state.processed_img = None
                    st.session_state.ai_response = None
                    st.session_state.confidence = 0
                    st.session_state.current_round = 0
                    st.rerun()
            with col_export:
                st.download_button(
                    label="💾 导出诊断报告",
                    data=report_text,
                    file_name="breadboard_diagnosis.md",
                    mime="text/markdown"
                )

# 侧边栏：增加置信度进度条
with st.sidebar:
    st.header("🧠 AI 协同思考轨迹")
    
    # 显示当前置信度进度条
    st.metric(label="当前推理置信度", value=f"{st.session_state.confidence}%")
    st.progress(st.session_state.confidence / 100.0)
    
    st.divider()
    if st.session_state.sidebar_log:
        for log in st.session_state.sidebar_log:
            st.markdown(f"• {log}")
    else:
        st.write("📝 暂无诊断记录。上传图片并启动诊断后，此处将实时显示 AI 推理摘要。")
    
    st.divider()
    st.caption("💡 设计原则：AI 仅对清晰可见部分做高置信度判断，模糊区域必标存疑并给出万用表验证建议，杜绝盲目断言。")
