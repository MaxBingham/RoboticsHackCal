import time
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from config import ELEVENLABS_API_KEY, AGENT_ID

def pick_object(params):
    color = params.get("color")
    print("\n" + "="*40)
    print(f"  ERKANNT: {color.upper()}")
    print("="*40 + "\n")
    return {"status": "success", "message": f"{color} Objekt aufgehoben"}

client_tools = ClientTools()
client_tools.register("pick_object", pick_object, is_async=False)

elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

conversation = Conversation(
    client=elevenlabs,
    agent_id=AGENT_ID,
    requires_auth=True,
    audio_interface=DefaultAudioInterface(),
    client_tools=client_tools,
)

print("Starte Gespräch... sprich einfach los, z.B. 'heb das gelbe auf'. Strg+C zum Beenden.")
conversation.start_session()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nBeende Gespräch...")
    conversation.end_session()
    conversation.wait_for_session_end()