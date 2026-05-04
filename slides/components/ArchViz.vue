<script setup>
import { ref, computed } from 'vue'

const active = ref(null)

function select(id) {
  active.value = active.value === id ? null : id
}

const details = {
  ppg: {
    title: 'PPG Downsampler',
    color: '#c0524a',
    io: '[B, 15,000]  →  [B, 500, 128]',
    params: '49,952',
    layers: [
      ['Conv1d(1→32, k=30, stride=5)', '[B, 32, 3,000]'],
      ['GELU + BatchNorm1d', ''],
      ['Conv1d(32→128, k=6, stride=6)', '[B, 128, 500]'],
      ['GELU + BatchNorm1d', ''],
      ['Transpose(1, 2)', '[B, 500, 128]'],
    ],
    note: '30× compression — retains ~4 Hz resolution from 125 Hz raw signal. Two-stage design: first widens channels (1→32), then downsamples aggressively with stride (32→128, 500 tokens).',
  },
  poincare: {
    title: 'Poincaré Sequence Encoder',
    color: '#6aabff',
    io: '4 × [B, 32, 32]  →  [B, 4, 128]',
    params: '144,256',
    layers: [
      ['PoincareCNN  (weights shared across 4 plots)', ''],
      ['  Conv2d(1→16, k=3) → ReLU → MaxPool(2)', '[B×4, 16, 15, 15]'],
      ['  Conv2d(16→32, k=3) → ReLU → MaxPool(2)', '[B×4, 32, 6, 6]'],
      ['  Flatten → Linear(1152→64)', '[B×4, 64]'],
      ['Linear(64→128)  applied per plot', '[B, 4, 128]'],
    ],
    note: 'Shared CNN weights mean the model learns one universal HRV density extractor. Each of the 4 Poincaré plots (one per 30s sub-window) becomes one token, appended to the PPG sequence.',
  },
  fusion: {
    title: 'Fusion + Positional Encoding',
    color: '#f0c060',
    io: '[B, 500, 128]  ⊕  [B, 4, 128]  →  [B, 504, 128]',
    params: '0 learned',
    layers: [
      ['cat([ppg_tokens, poincaré_tokens], dim=1)', '[B, 504, 128]'],
      ['PositionalEncoding  (sinusoidal, d=128)', '[B, 504, 128]'],
      ['Dropout(p=0.1)', ''],
    ],
    note: 'Poincaré tokens sit at positions 500–503. The transformer attends across all 504 positions jointly — it can learn cross-modal relationships between any PPG time step and any HRV summary.',
  },
  transformer: {
    title: 'Transformer Encoder  ×4',
    color: '#a78bfa',
    io: '[B, 504, 128]  →  [B, 504, 128]',
    params: '788,992',
    layers: [
      ['MultiHeadAttention(d_model=128, nhead=8)', '65,536 params / layer'],
      ['LayerNorm + Dropout', ''],
      ['FFN: Linear(128→512) → GELU → Linear(512→128)', '131,712 params / layer'],
      ['LayerNorm + Dropout', ''],
      ['× 4 stacked layers', '788,992 total'],
    ],
    note: 'Each of 8 heads attends over a 16-dim subspace. The 4× FFN expansion is the standard transformer ratio. Self-attention is the only operation that can relate distant time steps — essential for Mayer wave encoding.',
  },
  head: {
    title: 'Output Head',
    color: '#4ade80',
    io: '[B, 504, 128]  →  [B, 2]',
    params: '258',
    layers: [
      ['Mean pool over sequence dim  (504 → 1)', '[B, 128]'],
      ['Linear(128 → 2)', '[B, 2]'],
      ['Output: [SBP, DBP]', 'mmHg'],
    ],
    note: 'Global mean pooling gives every token equal contribution to the final prediction. A single linear layer jointly regresses SBP and DBP under one MSE loss — no separate heads.',
  },
}

const current = computed(() => active.value ? details[active.value] : null)
const currentColor = computed(() => current.value?.color ?? '#c0524a')
</script>

<template>
  <div style="display:flex;flex-direction:column;gap:6px;max-height:430px;overflow:hidden">

    <!-- Architecture diagram -->
    <svg viewBox="0 0 920 185" style="width:100%;flex-shrink:0;max-height:160px">
      <defs>
        <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0.5 L0,5.5 L5.5,3 z" fill="#555"/>
        </marker>
      </defs>

      <!-- ── Static input nodes ───────────────────────────── -->
      <rect x="5" y="8" width="112" height="46" rx="5" fill="#0e0e0e" stroke="#3a3a3a" stroke-width="1"/>
      <text x="61" y="27" text-anchor="middle" style="font-size:9px;font-family:monospace" fill="#999">PPG Input</text>
      <text x="61" y="41" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#555">125 Hz · 2 min</text>
      <text x="61" y="51" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">15,000 samples</text>

      <rect x="5" y="130" width="112" height="46" rx="5" fill="#0e0e0e" stroke="#3a3a3a" stroke-width="1"/>
      <text x="61" y="149" text-anchor="middle" style="font-size:9px;font-family:monospace" fill="#999">Poincaré Plots</text>
      <text x="61" y="163" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#555">4 × 30s windows</text>
      <text x="61" y="173" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">4 × 32×32</text>

      <!-- ── Input → clickable node arrows ─────────────────── -->
      <line x1="117" y1="31" x2="145" y2="31" stroke="#444" stroke-width="1.2" marker-end="url(#arr)"/>
      <line x1="117" y1="153" x2="145" y2="153" stroke="#444" stroke-width="1.2" marker-end="url(#arr)"/>

      <!-- ── PPG Downsampler ─────────────────────────────── -->
      <g @click="select('ppg')" style="cursor:pointer">
        <rect x="148" y="8" width="128" height="46" rx="5"
          :fill="active === 'ppg' ? '#1e0f0f' : '#0e0e0e'"
          :stroke="active === 'ppg' ? '#c0524a' : '#555'"
          stroke-width="1.5"/>
        <text x="212" y="26" text-anchor="middle" style="font-size:9px;font-family:monospace"
          :fill="active === 'ppg' ? '#c0524a' : '#ccc'">PPGDownsampler</text>
        <text x="212" y="39" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#666">Conv1d ×2</text>
        <text x="212" y="50" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">49,952 params</text>
      </g>

      <!-- ── Poincaré Encoder ───────────────────────────── -->
      <g @click="select('poincare')" style="cursor:pointer">
        <rect x="148" y="130" width="128" height="46" rx="5"
          :fill="active === 'poincare' ? '#0a1525' : '#0e0e0e'"
          :stroke="active === 'poincare' ? '#6aabff' : '#555'"
          stroke-width="1.5"/>
        <text x="212" y="148" text-anchor="middle" style="font-size:9px;font-family:monospace"
          :fill="active === 'poincare' ? '#6aabff' : '#ccc'">Poincaré Encoder</text>
        <text x="212" y="161" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#666">CNN + Linear</text>
        <text x="212" y="172" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">144,256 params</text>
      </g>

      <!-- ── Merge curves → Fusion ──────────────────────── -->
      <path d="M 276,31 C 312,31 314,82 340,82" fill="none" stroke="#444" stroke-width="1.2" marker-end="url(#arr)"/>
      <path d="M 276,153 C 312,153 314,102 340,102" fill="none" stroke="#444" stroke-width="1.2" marker-end="url(#arr)"/>
      <text x="292" y="24" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#444">[B,500,128]</text>
      <text x="292" y="170" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#444">[B,4,128]</text>

      <!-- ── Fusion + PE ─────────────────────────────────── -->
      <g @click="select('fusion')" style="cursor:pointer">
        <rect x="343" y="69" width="115" height="46" rx="5"
          :fill="active === 'fusion' ? '#1a1700' : '#0e0e0e'"
          :stroke="active === 'fusion' ? '#f0c060' : '#555'"
          stroke-width="1.5"/>
        <text x="400" y="87" text-anchor="middle" style="font-size:9px;font-family:monospace"
          :fill="active === 'fusion' ? '#f0c060' : '#ccc'">Fusion + PE</text>
        <text x="400" y="100" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#666">504 tokens</text>
        <text x="400" y="111" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">0 learned params</text>
      </g>

      <!-- ── Arrow: Fusion → Transformer ───────────────── -->
      <line x1="458" y1="92" x2="484" y2="92" stroke="#444" stroke-width="1.2" marker-end="url(#arr)"/>
      <text x="471" y="86" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#444">[B,504,128]</text>

      <!-- ── Transformer ×4 ─────────────────────────────── -->
      <g @click="select('transformer')" style="cursor:pointer">
        <rect x="487" y="69" width="132" height="46" rx="5"
          :fill="active === 'transformer' ? '#14102a' : '#0e0e0e'"
          :stroke="active === 'transformer' ? '#a78bfa' : '#555'"
          stroke-width="1.5"/>
        <text x="553" y="87" text-anchor="middle" style="font-size:9px;font-family:monospace"
          :fill="active === 'transformer' ? '#a78bfa' : '#ccc'">Transformer  ×4</text>
        <text x="553" y="100" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#666">MHA + FFN · d=128</text>
        <text x="553" y="111" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">788,992 params</text>
      </g>

      <!-- ── Arrow: Transformer → Head ─────────────────── -->
      <line x1="619" y1="92" x2="647" y2="92" stroke="#444" stroke-width="1.2" marker-end="url(#arr)"/>
      <text x="633" y="86" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#444">[B,504,128]</text>

      <!-- ── Output Head ────────────────────────────────── -->
      <g @click="select('head')" style="cursor:pointer">
        <rect x="650" y="69" width="108" height="46" rx="5"
          :fill="active === 'head' ? '#0c1a10' : '#0e0e0e'"
          :stroke="active === 'head' ? '#4ade80' : '#555'"
          stroke-width="1.5"/>
        <text x="704" y="87" text-anchor="middle" style="font-size:9px;font-family:monospace"
          :fill="active === 'head' ? '#4ade80' : '#ccc'">Output Head</text>
        <text x="704" y="100" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#666">MeanPool + Linear</text>
        <text x="704" y="111" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">258 params</text>
      </g>

      <!-- ── Arrow: Head → Output ───────────────────────── -->
      <line x1="758" y1="92" x2="784" y2="92" stroke="#444" stroke-width="1.2" marker-end="url(#arr)"/>
      <text x="771" y="86" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#444">[B,2]</text>

      <!-- ── Static output node ─────────────────────────── -->
      <rect x="787" y="69" width="128" height="46" rx="5" fill="#0e0e0e" stroke="#3a3a3a" stroke-width="1"/>
      <text x="851" y="87" text-anchor="middle" style="font-size:10px;font-family:monospace" fill="#c0524a">SBP / DBP</text>
      <text x="851" y="101" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#555">[B, 2]  ·  mmHg</text>
      <text x="851" y="111" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">985,506 total params</text>
    </svg>

    <!-- Detail panel / placeholder -->
    <Transition name="panel">
      <div v-if="current" key="panel"
        style="flex:1;border-radius:6px;padding:7px 12px;overflow:hidden;background:rgba(14,14,14,0.95)"
        :style="{ borderLeft: `3px solid ${currentColor}` }">
        <div style="display:flex;gap:18px;height:100%">

          <!-- Layers column -->
          <div style="flex:1.3;display:flex;flex-direction:column;gap:3px">
            <div style="display:flex;justify-content:space-between;align-items:baseline">
              <span style="font-family:monospace;font-size:10px;font-weight:bold" :style="{ color: currentColor }">
                {{ current.title }}
              </span>
              <span style="font-family:monospace;font-size:7px;color:#555">{{ current.params }} params</span>
            </div>
            <div style="font-family:monospace;font-size:7px;color:#666;margin-bottom:1px">{{ current.io }}</div>
            <div style="display:flex;flex-direction:column;gap:1px">
              <div v-for="([name, shape], i) in current.layers" :key="i"
                style="display:flex;justify-content:space-between;padding:2px 7px;background:#151515;border-radius:3px">
                <span style="font-family:monospace;font-size:7.5px;color:#bbb">{{ name }}</span>
                <span v-if="shape" style="font-family:monospace;font-size:7.5px;color:#555;padding-left:10px;white-space:nowrap">{{ shape }}</span>
              </div>
            </div>
          </div>

          <!-- Note column -->
          <div style="flex:1;border-left:1px solid #222;padding-left:12px;display:flex;align-items:center">
            <p style="font-family:monospace;font-size:8px;color:#888;line-height:1.6;margin:0">{{ current.note }}</p>
          </div>
        </div>
      </div>

      <div v-else key="hint"
        style="flex:1;display:flex;align-items:center;justify-content:center">
        <span style="font-family:monospace;font-size:8px;color:#333">click any block to expand</span>
      </div>
    </Transition>

  </div>
</template>

<style scoped>
.panel-enter-active,
.panel-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}
.panel-enter-from,
.panel-leave-to {
  opacity: 0;
  transform: translateY(6px);
}
</style>
