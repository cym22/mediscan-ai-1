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

// 语言配置
const LANGUAGE_CONFIG = {
  zh: { name: '中文', voiceName: 'Kore' },
  en: { name: 'English', voiceName: 'Kore' },
  ja: { name: '日本語', voiceName: 'Kore' },
  ko: { name: '한국어', voiceName: 'Kore' },
  ru: { name: 'Русский', voiceName: 'Kore' }
};

// 根据语言生成系统提示词
const getSystemPrompt = (mode, lang = 'zh') => {
  const langName = LANGUAGE_CONFIG[lang]?.name || '中文';
  
  if (mode === 'report') {
    return `你是一位经验丰富、和蔼可亲的全科医生。你的任务是帮助老年人看懂他们的体检报告。
请用最简单、最口语化的${langName}，像对待自己的父母一样解释报告内容。
首先识别图片/PDF中的所有文字内容，然后分析并按要求格式输出。
**严格按照以下JSON格式输出，不要输出任何其他内容：**
{
  "exam_date": "从报告中提取的检查日期，格式如 2024-01-15",
  "overall_summary": "用2-3句最简单的话概括整体情况，要让老人听了安心或重视",
  "good_news": ["好消息1，要具体", "好消息2"],
  "attention_needed": [
    {
      "item": "指标名称（如：空腹血糖）",
      "value": "具体数值和单位（如：7.2 mmol/L）",
      "explanation": "用最简单的话解释这是什么、为什么要关注",
      "advice": "具体、可操作的建议",
      "severity": "low/medium/high",
      "follow_up": {
        "timeline": "如：3个月后",
        "target_date": "根据检查日期计算的具体日期，格式如 2024-04-15",
        "action": "需要做什么检查"
      }
    }
  ],
  "diet_lifestyle_guide": ["具体的饮食建议1", "具体的生活习惯建议2"]
}`;
  } else if (mode === 'medicine') {
    return `你是一位经验丰富的药剂师，正在帮助一位老人理解他手中的药物。
请仔细查看药盒或说明书的照片，用最简单、最口语化的${langName}解释。
首先识别图片中的所有文字内容，然后分析并按要求格式输出。
**严格按照以下JSON格式输出：**
{
  "name": "药品名称（通用名和商品名）",
  "efficacy": "这个药是治什么的，用一句话说清楚",
  "usage": "怎么吃、吃多少、什么时候吃，要非常具体",
  "contraindications": "什么情况不能吃，用口语说",
  "side_effects_alert": "可能会有什么反应，哪些反应需要注意",
  "summary": "用一句话总结最重要的注意事项"
}`;
  } else if (mode === 'food') {
    return `你是一位关心老年人健康的营养师。
请查看这个食品的配料表和营养成分表，帮老人判断这个食品适不适合吃。
首先识别图片中的所有文字内容，然后分析并按要求格式输出。
用${langName}回复。
**严格按照以下JSON格式输出：**
{
  "name": "食品名称",
  "ingredients_analysis": "用简单的话说说主要成分是什么",
  "additives_alert": ["需要注意的添加剂1", "添加剂2（如果有的话）"],
  "nutrition_alert": {
    "sugar": "low/medium/high",
    "salt": "low/medium/high",
    "fat": "low/medium/high"
  },
  "advice_for_elderly": "针对老年人的具体建议，比如有糖尿病能不能吃",
  "summary": "一句话总结：推荐/适量/不推荐，为什么"
}`;
  }
  return '';
};

// 分析接口
app.post('/api/analyze', async (req, res) => {
  try {
    const { mode, images, pdfData, language = 'zh' } = req.body;
    if (!mode || (!images?.length && !pdfData)) {
      return res.status(400).json({ error: '缺少必要参数' });
    }
    
    const ai = getAiClient();
    const systemPrompt = getSystemPrompt(mode, language);
    
    const parts = [];
    if (images?.length > 0) {
      for (const img of images) {
        parts.push({ inlineData: { mimeType: img.mimeType || 'image/jpeg', data: img.base64 } });
      }
    }
    if (pdfData) {
      parts.push({ inlineData: { mimeType: 'application/pdf', data: pdfData.base64 } });
    }
    parts.push({ text: '请先识别图片/PDF中的所有文字，然后分析并按JSON格式输出。' });
    
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

// TTS接口 - 优化版
app.post('/api/tts', async (req, res) => {
  try {
    const { text, language = 'zh' } = req.body;
    if (!text) return res.status(400).json({ error: '缺少文本' });
    
    // 限制文本长度，避免生成时间过长
    const maxLength = 500;
    const truncatedText = text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    
    const ai = getAiClient();
    const voiceName = LANGUAGE_CONFIG[language]?.voiceName || 'Kore';
    
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-preview-tts',
      contents: [{ parts: [{ text: truncatedText }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName } }
        }
      }
    });
    
    const audioData = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!audioData) return res.status(500).json({ error: '语音生成失败' });
    
    res.json({ success: true, audio: audioData });
  } catch (error) {
    console.error('TTS error:', error);
    res.status(500).json({ error: '语音生成失败', details: error.message });
  }
});

// 对话接口
app.post('/api/chat', async (req, res) => {
  try {
    const { message, contextType, contextItem, contextContent, history, language = 'zh' } = req.body;
    if (!message) return res.status(400).json({ error: '缺少消息' });
    
    const ai = getAiClient();
    const langName = LANGUAGE_CONFIG[language]?.name || '中文';
    
    const systemPrompt = `你是一位耐心、专业的健康顾问，正在帮助一位老年用户理解他们的${contextType || '健康信息'}。
当前讨论的是：${contextItem || '健康问题'}
背景信息：${contextContent || '无'}

请用简单、口语化的${langName}回答问题。回答要简洁明了，避免使用专业术语。
如果用户的问题涉及需要就医的情况，请明确建议他们去看医生。`;

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
