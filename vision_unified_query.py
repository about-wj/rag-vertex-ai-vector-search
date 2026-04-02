import os
import re
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

model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
llm = GenerativeModel("gemini-2.5-flash") 
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=ENDPOINT_NAME)

def extract_image_path(text):
    """使用正则自动从用户输入中提取本地图片路径"""
    # 匹配 Linux 或 Windows 下常见的图片路径格式
    match = re.search(r'([a-zA-Z0-9_./\\-]+\.(?:jpg|jpeg|png))', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def unified_vision_rag_chat():
    while True:
        # 用户可以直接输入包含路径的整句话
        user_input = input("\n👉 请提问 (例如: 请看附图的型号 /path/test.jpg) (输入 q 退出):\n> ")
        
        if user_input.lower() == 'q':
            break

        # 1. 智能解析用户输入
        image_path = extract_image_path(user_input)
        has_local_image = False
        
        if image_path and os.path.exists(image_path):
            has_local_image = True
            # 从原始提问中把路径字符串删掉，剩下的就是纯净的 Prompt
            clean_prompt = user_input.replace(image_path, "").strip()
            print(f"🔍 检测到本地图片: {image_path}")
        else:
            clean_prompt = user_input
            if image_path:
                print(f"⚠️ 检测到了路径 {image_path}，但本地文件不存在，将降级为纯文本搜索。")

        # 2. 生成检索向量
        print("1. 正在生成多模态检索向量...")
        try:
            if has_local_image:
                v_image = VisionImage.load_from_file(image_path)
                embeddings = model.get_embeddings(
                    image=v_image, 
                    contextual_text=clean_prompt if clean_prompt else None
                )
                query_vector = embeddings.image_embedding
            else:
                # 纯文本检索
                query_vector = model.get_embeddings(contextual_text=clean_prompt).text_embedding
        except Exception as e:
            print(f"❌ 向量化失败: {str(e)}")
            continue

        # 3. 向量数据库检索
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
            print(f"✅ 检索命中文档页面 ID: {best_match_id} (相似度: {distance:.4f})")
            
        except Exception as e:
            print(f"❌ 检索失败: {str(e)}")
            continue

        retrieved_image_path = f"pdf_pages/{best_match_id}.jpg"
        
        # 4. 构建大模型 Prompt 并生成回答
        print("3. Gemini 2.5 flash 正在思考中...")
        try:
            doc_img_part = Part.from_image(GenAIImage.load_from_file(retrieved_image_path))
            
            if has_local_image:
                user_img_part = Part.from_image(GenAIImage.load_from_file(image_path))
                prompt_text = (
                    f"你是一个强大的技术支持专家。\n"
                    f"用户上传了一张照片（图1），并在官方文档中匹配到了相关页面（图2）。\n"
                    f"请仔细查看两张图片，并回答用户的问题：\n{clean_prompt}"
                )
                # 送入：用户图 + 文档图 + Prompt
                answer = llm.generate_content([user_img_part, doc_img_part, prompt_text])
            else:
                prompt_text = f"请仔细观察这张文档页面的原始截图。基于这张截图的内容，回答以下问题：\n{clean_prompt}"
                # 仅送入：文档图 + Prompt
                answer = llm.generate_content([doc_img_part, prompt_text])

            print("\n🤖 Gemini 回答：\n" + "="*40)
            print(answer.text)
            print("=" * 40)
        except Exception as e:
            print(f"❌ 生成回答时发生错误: {str(e)}")

if __name__ == "__main__":
    unified_vision_rag_chat()
