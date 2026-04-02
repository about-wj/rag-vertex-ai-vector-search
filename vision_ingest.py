import os
from pdf2image import convert_from_path
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import IndexDatapoint
import vertexai
from vertexai.vision_models import Image as VisionImage
from vertexai.vision_models import MultiModalEmbeddingModel

# ================= 配置区 =================
PROJECT_ID = "your-gcp-project-id"  # 替换为你真实的 Project ID
LOCATION = "us-central1"
# 填入你新创建的 1408 维 Index 的完整路径，形如：projects/.../locations/.../indexes/...
INDEX_NAME = "your-index-id" 
PDF_PATH = "/path/xxxxxxxxxx.pdf" # 你本地的 PDF 文件路径
# ==========================================

print("正在初始化 GCP 环境...")
vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# 加载多模态 Embedding 模型 (输出 1408 维)
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

print(f"正在将 PDF ({PDF_PATH}) 转换为图像...")
# 注意：Windows 用户如果 poppler 环境变量没配好，可以在这里指定 poppler_path=r'C:\poppler\bin'
pages = convert_from_path(PDF_PATH, dpi=150)

datapoints_to_insert =[]

# 创建一个目录用来存放图片
os.makedirs("pdf_pages", exist_ok=True)

for i, page in enumerate(pages):
    image_path = f"pdf_pages/page_{i}.jpg"
    page.save(image_path, "JPEG")
    print(f"已保存并正在向量化: {image_path}")
    
    # 1. 获取整张图片的 1408 维特征向量
    v_image = VisionImage.load_from_file(image_path)
    embeddings = model.get_embeddings(image=v_image)
    image_vector = embeddings.image_embedding
    
    # 2. 组装成 Vector Search 需要的数据格式
    # 注意：这里修正了你的原代码，使用了正确的 IndexDatapoint 类
    dp = IndexDatapoint(
        datapoint_id=f"page_{i}",
        feature_vector=image_vector,
        restricts=[IndexDatapoint.Restriction(namespace="page_number", allow_list=[str(i)])]
    )
    datapoints_to_insert.append(dp)

print(f"准备将 {len(datapoints_to_insert)} 页图片向量推送到 Vector Search...")
# 3. 写入数据（修正：应该调用 Index 实例，而不是 Endpoint 实例来 upsert）
my_index = aiplatform.MatchingEngineIndex(index_name=INDEX_NAME)
my_index.upsert_datapoints(datapoints=datapoints_to_insert)

print("图片向量入库成功！")
