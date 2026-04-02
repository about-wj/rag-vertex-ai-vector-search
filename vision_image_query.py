import os
from google.cloud import aiplatform
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image as VisionImage
from vertexai.generative_models import GenerativeModel, Part, Image as GenAIImage

# ================= 1. 配置区 =================
PROJECT_ID = "bd-host-2026-002"
LOCATION = "us-central1"
ENDPOINT_NAME = "5517589041714823168"
DEPLOYED_INDEX_ID = "gemini_vector_search_deployed"
# ============================================

print("正在初始化环境...")
vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# 加载模型
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
llm = GenerativeModel("gemini-1.5-pro-002") 
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=ENDPOINT_NAME)

def vision_image_rag_chat():
    while True:
        # 1. 接收用户的图片路径
        user_image_path = input("\n📸 请输入要搜索的产品图片路径 (例如 ./my_product.jpg, 输入 q 退出): ")
        if user_image_path.lower() == 'q':
            break
        if not os.path.exists(user_image_path):
            print("❌ 找不到该图片，请检查路径。")
            continue

        # 2. 接收用户的附加问题 (可选)
        user_query = input("👉 你想问关于这个产品的什么问题？ (直接回车默认查询: 请详细介绍一下该产品的规格和参数): ")
        if not user_query.strip():
            user_query = "请根据文档手册页面，详细介绍一下该产品的规格、特点和参数。"

        print("1. 正在将您的 [实拍图片] 转换为向量...")
        try:
            # 【核心变化】：将图片加载并转换为向量
            v_image = VisionImage.load_from_file(user_image_path)
            query_vector = model.get_embeddings(image=v_image).image_embedding
        except Exception as e:
            print(f"❌ 图片向量化失败: {str(e)}")
            continue

        print("2. 正在向量数据库中搜寻最相关的文档页面...")
        try:
            response = my_index_endpoint.find_neighbors(
                deployed_index_id=DEPLOYED_INDEX_ID,
                queries=[query_vector],
                num_neighbors=1 
            )
            
            neighbors = response[0].neighbors if hasattr(response[0], 'neighbors') else response[0]
            
            if not neighbors:
                print("❌ 未能检索到相关页面。")
                continue
                
            best_match_id = neighbors[0].id
            distance = neighbors[0].distance
            print(f"✅ 检索命中产品文档页面 ID: {best_match_id} (相似度: {distance:.4f})")
            
        except Exception as e:
            print(f"❌ 检索失败: {str(e)}")
            continue

        retrieved_image_path = f"pdf_pages/{best_match_id}.jpg"
        
        try:
            # 将用户的图片和检索到的文档图片都加载进来
            user_img_part = Part.from_image(GenAIImage.load_from_file(user_image_path))
            retrieved_img_part = Part.from_image(GenAIImage.load_from_file(retrieved_image_path))
        except FileNotFoundError:
            print(f"❌ 找不到需要加载的图片文件。")
            continue

        # 【核心变化】：调整 Prompt，让 Gemini 明白这两张图的关系
        prompt = (
            f"你是一个强大的产品技术支持专家。\n"
            f"用户提供了一张产品的实物图（图1），我们从官方产品手册中检索到了最相关的说明书页面（图2）。\n"
            f"请仔细核对图2（说明书）的内容，并回答用户关于图1（实物产品）的问题：\n\n"
            f"用户问题：{user_query}"
        )
        
        print("3. Gemini 1.5 Pro 正在双图对比思考中...")
        try:
            # 同时将 【用户图】、【文档图】和【提示词】喂给大模型
            answer = llm.generate_content([user_img_part, retrieved_img_part, prompt])
            print("\n🤖 Gemini 专家回答：\n" + "="*40)
            print(answer.text)
            print("=" * 40)
        except Exception as e:
            print(f"❌ 生成回答时发生错误: {str(e)}")

if __name__ == "__main__":
    vision_image_rag_chat()
