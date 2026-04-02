from google.cloud import aiplatform
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel
# 注意：生成模型使用的 Image 类和 Embedding 使用的稍有不同
from vertexai.generative_models import GenerativeModel, Part, Image as GenAIImage

# ================= 配置区 =================
PROJECT_ID = "bd-host-2026-002"
LOCATION = "us-central1"
# 填入你新部署的 Endpoint 完整路径
ENDPOINT_NAME = "5517589041714823168"
DEPLOYED_INDEX_ID = "gemini_vector_search_deployed" # 部署时设置的名字
# ==========================================

vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=ENDPOINT_NAME)
llm = GenerativeModel("gemini-2.5-flash")

def vision_rag_chat():
    while True:
        user_query = input("\n👉 请输入针对 PDF 截图的问题 (输入 q 退出): ")
        if user_query.lower() == 'q':
            break

        print("1. 正在将您的问题转换为向量...")
        query_embeddings = model.get_embeddings(contextual_text=user_query)
        query_vector = query_embeddings.text_embedding

        print("2. 正在向量数据库中搜寻最相关的原图...")
        response = my_index_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_vector],
            num_neighbors=1 # 找到最相关的那一页
        )

        # 修正：处理 Vector Search SDK 的返回格式
        neighbors = response[0].neighbors if hasattr(response[0], 'neighbors') else response[0]
        if not neighbors:
            print("❌ 未能检索到相关页面。")
            continue
                
        best_match_id = neighbors[0].id
        distance = neighbors[0].distance
        print(f"✅ 检索命中页面 ID: {best_match_id} (相似度: {distance:.4f})")

        # 3. 将本地保存的对应原图捞出来
        retrieved_image_path = f"pdf_pages/{best_match_id}.jpg"
        
        try:
            image_part = Part.from_image(GenAIImage.load_from_file(retrieved_image_path))
        except FileNotFoundError:
            print(f"❌ 找不到本地图片文件: {retrieved_image_path}")
            continue

        # 4. 把【原图】和【问题】一起喂给 Gemini 2.5 flash
        prompt = f"请仔细观察这张文档页面的原始截图。基于这张截图的内容，回答以下问题：\n{user_query}"
        
        print("3. Gemini 2.5 flash 正在看图思考中...")
        try:
            answer = llm.generate_content([image_part, prompt])
            print("\n🤖 Gemini 回答：\n" + "-"*40)
            print(answer.text)
            print("-" * 40)
        except Exception as e:
            print(f"生成回答时发生错误: {str(e)}")

if __name__ == "__main__":
    vision_rag_chat()
