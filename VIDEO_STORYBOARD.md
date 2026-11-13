# Demo Video — Production Plan & Storyboard

**Formative 2 — Multimodal Data Preprocessing**  
**Prepared by:** Member 4  
**Purpose:** System simulation video for the assignment submission

---

## Objective

Produce a 2–4 minute video demonstrating the complete User Identity & Product
Recommendation System flow, including both the authorised transaction and the
unauthorised-attempt denial pathways.

---

## Prerequisites

Before recording, ensure:

1. The repository is cloned and all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. The CLI app runs successfully:
   ```bash
   python system_app.py
   ```
3. All three models are present in `models/`:
   - `product_model.joblib`
   - `face_model.joblib`
   - `voice_model.joblib`

---

## Equipment & Software

- **Screen recorder:** OBS Studio (free), Loom, QuickTime Player (Mac), or
  Windows Game Bar (`Win + G`)
- **Microphone:** Any USB or headset microphone (built-in laptop mic is acceptable
  in a quiet room)
- **Display:** Set terminal/font size large enough to be readable on video
  (recommend 14–16 pt terminal font)

---

## Storyboard (7 Scenes)

| # | Scene | Duration | Action | Narration |
|---|-------|----------|--------|-----------|
| 1 | **Intro** | ~20s | Open terminal in the repo root. Show `ls` to confirm `system_app.py` exists. | "Welcome to the User Identity and Product Recommendation System demo. I'll show how a user authenticates with face and voice before receiving a product recommendation." |
| 2 | **Launch** | ~15s | Run `python system_app.py`. Wait for "Models loaded" message. | "First, we launch the CLI application. It loads three models: face recognition, voiceprint verification, and product recommendation." |
| 3 | **Demo 1 start** | ~30s | The app presents Demo 1: face image + voice sample for an enrolled member (e.g., mukasa). | "Demo 1 is a full authorised transaction. We present Mukasa's face image and voice sample." |
| 4 | **Demo 1 pass** | ~30s | Terminal shows face recognised, voice verified, multimodal gate passed, product category recommended (e.g., Sports). | "The face recognition model identifies Mukasa, the voiceprint model verifies the voice, both checks pass, and the system grants access with a recommended product category." |
| 5 | **Demo 2 start** | ~20s | The app presents Demo 2: unauthorized attempt. | "Demo 2 shows what happens when an impostor tries to access the system." |
| 6 | **Demo 2 deny** | ~30s | Terminal shows either face not recognised OR voice not verified, followed by ACCESS DENIED. | "The impostor's face or voice fails verification. The multimodal gate denies access immediately — no product recommendation is served." |
| 7 | **Close** | ~15s | Summarise the flow; show that verified users only get recommendations. | "In summary: face plus voice must both pass. One failure means denied access. Only verified users receive product recommendations." |

**Total target length: 2–3 minutes.**

---

## Recording Tips

1. **Font size:** Increase terminal font size before recording so text is legible
   on smaller screens.
2. **Narration:** Speak clearly, at a moderate pace. Narrate what is happening on
   screen as it happens.
3. **Quiet environment:** Close windows, turn off fans/music, silence phone
   notifications.
4. **Single take preferred:** Record continuously from scene 1 to scene 7. If
   you make a mistake, pause briefly and continue — you can edit later.
5. **Show the code (optional):** If time permits, briefly open `system_app.py`
   to show the multimodal gate logic (`face_ok` AND `voice_ok` required).

---

## Submission

1. Save the video as an MP4 file.
2. Upload to a shareable location (YouTube unlisted, Google Drive, OneDrive,
   Stream).
3. Paste the link into the team's final report under the "Demo Video" section.
4. Include the link in `README.md` if appropriate.
