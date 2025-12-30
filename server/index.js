import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Modality } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, '../public')));

const getAiClient = () => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY is not set');
  return new GoogleGenAI({ apiKey });
};

// 分析接口
app.post('/api/analyze', async (req, res) => {
  try {
    const { mode, images, pdfData } = req.body;
    if (!mode || (!images?.length && !pdfData)) {
      return res.status(400).json({ error: '缺少必要参数' });
    }
    const ai = getAiClient();
    let systemPrompt = '';
    if (mode === 'report') {
      systemPrompt = `你是一位经验丰富、和蔼可亲的全科医生。用最简单的语言帮老年人看懂体检报告。严格按JSON格式输出：{"exam_date":"检查日期","overall_summary":"2-3句话概括","good_news":["好消息1"],"attention_needed":[{"item":"指标名","value":"数值","explanation":"简单解释","advice":"建议","severity":"low/medium/high","follow_up":{"timeline":"如3个月后","target_date":"具体日期","action":"做什么检查"}}],"diet_lifestyle_guide":["建议1"]}`;
    } else if (mode === 'medicine') {
      systemPrompt = `你是一位药剂师，帮老人理解药物。严格按JSON格式输出：{"name":"药品名","efficacy":"治什么的","usage":"怎么吃","contraindications":"什么情况不能吃","side_effects_alert":"可能的副作用","summary":"最重要的注意事项"}`;
    } else if (mode === 'food') {
      systemPrompt = `你是一位营养师，帮老人看食品配料表。严格按JSON格式输出：{"name":"食品名","ingredients_analysis":"主要成分","additives_alert":["添加剂1"],"nutrition_alert":{"sugar":"low/medium/high","salt":"low/medium/high","fat":"low/medium/high"},"advice_for_elderly":"老人建议","summary":"一句话总结"}`;
    }
    const parts = [];
    if (images?.length > 0) {
      for (const img of images) {
        parts.push({ inlineData: { mimeType: img.mimeType || 'image/jpeg', data: img.base64 } });
      }
    }
    if (pdfData) {
      parts.push({ inlineData: { mimeType: 'application/pdf', data: pdfData.base64 } });
    }
    parts.push({ text: '请分析并按JSON格式输出。' });
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: [{ role: 'user', parts }],
      config: { systemInstruction: systemPrompt, temperature: 0.3 }
    });
    const text = response.candidates?.[0]?.content?.parts?.[0]?.text || '';
    let jsonStr = text;
    const jsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/) || text.match(/\{[\s\S]*\}/);
    if (jsonMatch) jsonStr = jsonMatch[1] || jsonMatch[0];
    res.json({ success: true, data: JSON.parse(jsonStr) });
  } catch (error) {
    console.error('Analyze error:', error);
    res.status(500).json({ error: '分析失败，请重试', details: error.message });
  }
});

// TTS接口
app.post('/api/tts', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: '缺少文本' });
    const ai = getAiClient();
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-preview-tts',
      contents: [{ parts: [{ text: `请朗读：${text}` }] }],
      config: { responseModalities: [Modality.AUDIO], speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } } }
    });
    const audioData = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!audioData) return res.status(500).json({ error: '语音生成失败' });
    res.json({ success: true, audio: audioData });
  } catch (error) {
    console.error('TTS error:', error);
    res.status(500).json({ error: '语音生成失败' });
  }
});

// 对话接口
app.post('/api/chat', async (req, res) => {
  try {
    const { message, contextType, contextItem, contextContent, history } = req.body;
    if (!message) return res.status(400).json({ error: '缺少消息' });
    const ai = getAiClient();
    const systemPrompt = `你是一位耐心的健康顾问，正在帮助老年用户理解${contextType || '健康信息'}。当前讨论：${contextItem || ''}。背景：${contextContent || '无'}。请用简单口语回答，避免专业术语。`;
    const contents = [];
    if (history?.length > 0) {
      for (const msg of history) {
        contents.push({ role: msg.role === 'user' ? 'user' : 'model', parts: [{ text: msg.text }] });
      }
    }
    contents.push({ role: 'user', parts: [{ text: message }] });
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents,
      config: { systemInstruction: systemPrompt, temperature: 0.7 }
    });
    res.json({ success: true, reply: response.candidates?.[0]?.content?.parts?.[0]?.text || '' });
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ error: '回复失败' });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', hasApiKey: !!process.env.GEMINI_API_KEY });
});

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/index.html'));
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
