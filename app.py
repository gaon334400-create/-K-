import streamlit as st
from google import genai
from google.genai import types

# 1. 페이지 및 레이아웃 설정
st.set_page_config(page_title="마음쉼터 - 심리상담 챗봇", page_icon="🌿", layout="centered")
st.title("🌿 마음쉼터 심리상담 AI")
st.caption("누구에게도 말하지 못했던 이야기, 편하게 들려주세요. 당신의 이야기를 경청합니다.")

# 2. Gemini 클라이언트 초기화 및 API 키 로드
# st.secrets를 통해 .streamlit/secrets.toml 에 저장된 키를 안전하게 가져옵니다.
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except KeyError:
    st.error(".streamlit/secrets.toml 파일에 'GEMINI_API_KEY'가 설정되지 않았습니다.")
    st.stop()
except Exception as e:
    st.error(f"클라이언트 초기화 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 심리 상담사 페르소나 정의 (System Instruction)
SYSTEM_INSTRUCTION = """
당신은 공감 능력이 뛰어나고 따뜻한 전문 심리 상담사입니다. 
사용자의 감정을 깊이 경청하고, 비난하거나 판단하지 않고 온전히 수용하는 태도로 대화해야 합니다.
조언을 성급하게 건네기보다는 사용자가 자신의 감정을 털어놓을 수 있도록 열린 질문을 던지며 공감해 주세요.
단, 자해, 자살, 혹은 심각한 정신적 위기 징후가 보일 때는 따뜻한 위로와 함께 반드시 전문 기관(예: 정신건강복지센터, 상담전화 등)의 도움을 받도록 안내해야 합니다.
답변은 친근하고 다정한 존댓말(해요체 등)을 사용해 주세요.
"""

# 4. 세션 상태(Session State)로 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. 과거 대화 내용 화면에 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. 사용자 입력 처리
if prompt := st.chat_input("지금 어떤 마음이 드시나요?"):
    
    # 사용자의 입력 화면 표시 및 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 챗봇(어시스턴트)의 응답 영역 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # SDK 규격에 맞게 대화 기록(History) 구성 변환
        contents = []
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            ))
        
        # 모델 설정 (요청하신 gemini-2.5-flash-lite 모델 및 페르소나 적용)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7, # 조금 더 자연스럽고 공감어린 표현을 위해 조정
        )
        
        try:
            # 실시간 스트리밍 방식으로 상담 답변 생성
            response_stream = client.models.generate_content_stream(
                model='gemini-2.5-flash-lite',
                contents=contents,
                config=config
            )
            
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    # 실시간으로 텍스트를 채워나가는 커서 효과
                    message_placeholder.markdown(full_response + "▌")
                    
            # 스트리밍 완료 후 커서 지우고 최종 텍스트 확정 출력
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            # API 호출 오류 처리
            st.error(f"상담봇과 연결하는 중에 오류가 발생했습니다. 잠시 후 다시 시도해 주세요. (에러: {e})")
            full_response = "죄송해요, 잠시 마음을 정리하는 데 시간이 걸리고 있어요. 다시 한번 이야기해 주실래요?"
            message_placeholder.markdown(full_response)
            
    # 대화 이력 유지 보관를 위해 챗봇 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})
