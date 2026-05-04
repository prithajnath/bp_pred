<template>
  <div style="display:flex;flex-direction:column;gap:6px;max-height:430px;overflow:hidden">

    <svg viewBox="0 0 920 185" style="width:100%;flex-shrink:0;max-height:180px">
      <defs>
        <marker id="arr2" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
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

      <!-- ── Input → node arrows ─────────────────────────── -->
      <line x1="117" y1="31" x2="145" y2="31" stroke="#444" stroke-width="1.2" marker-end="url(#arr2)"/>
      <line x1="117" y1="153" x2="145" y2="153" stroke="#444" stroke-width="1.2" marker-end="url(#arr2)"/>

      <!-- ── CNN Downsampler — fades out on click 2 ───────── -->
      <g :style="{ opacity: $clicks >= 2 ? 0 : 1, transition: 'opacity 0.45s ease' }">
        <rect x="148" y="8" width="128" height="46" rx="5" fill="#0e0e0e" stroke="#555" stroke-width="1.5"/>
        <text x="212" y="26" text-anchor="middle" style="font-size:9px;font-family:monospace" fill="#ccc">PPGDownsampler</text>
        <text x="212" y="39" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#666">Conv1d ×2</text>
        <text x="212" y="50" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">49,952 params</text>
      </g>

      <!-- ── Yellow highlight — click 1, fades out on click 2 ─ -->
      <g v-click :style="{ opacity: $clicks >= 2 ? 0 : 1, transition: 'opacity 0.3s ease' }">
        <rect x="145" y="5" width="134" height="52" rx="6" fill="rgba(240,192,96,0.07)" stroke="#f0c060" stroke-width="2"/>
      </g>

      <!-- ── PapaGei block — click 2 ───────────────────────── -->
      <g v-click>
        <rect x="148" y="8" width="128" height="46" rx="5" fill="#1a0e1f" stroke="#c084fc" stroke-width="1.8"/>
        <text x="212" y="24" text-anchor="middle" style="font-size:9px;font-family:monospace" fill="#c084fc">PapaGei</text>
        <text x="212" y="37" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#9d5fd4">Foundation Model</text>
        <text x="212" y="48" text-anchor="middle" style="font-size:7.5px;font-family:monospace" fill="#7c3aad">Embeddings</text>
        <rect x="252" y="9" width="22" height="11" rx="3" fill="#c084fc22" stroke="#c084fc" stroke-width="0.8"/>
        <text x="263" y="17" text-anchor="middle" style="font-size:6px;font-family:monospace;font-weight:bold" fill="#c084fc">NEW</text>
      </g>

      <!-- ── Poincaré Encoder (static) ─────────────────── -->
      <rect x="148" y="130" width="128" height="46" rx="5" fill="#0e0e0e" stroke="#555" stroke-width="1.5"/>
      <text x="212" y="148" text-anchor="middle" style="font-size:9px;font-family:monospace" fill="#ccc">Poincaré Encoder</text>
      <text x="212" y="161" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#666">CNN + Linear</text>
      <text x="212" y="172" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">144,256 params</text>

      <!-- ── Merge curves → Fusion ──────────────────────── -->
      <path d="M 276,31 C 312,31 314,82 340,82" fill="none" stroke="#444" stroke-width="1.2" marker-end="url(#arr2)"/>
      <path d="M 276,153 C 312,153 314,102 340,102" fill="none" stroke="#444" stroke-width="1.2" marker-end="url(#arr2)"/>
      <text x="292" y="24" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#444">[B,500,128]</text>
      <text x="292" y="170" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#444">[B,4,128]</text>

      <!-- ── Fusion + PE (static) ───────────────────────── -->
      <rect x="343" y="69" width="115" height="46" rx="5" fill="#0e0e0e" stroke="#555" stroke-width="1.5"/>
      <text x="400" y="87" text-anchor="middle" style="font-size:9px;font-family:monospace" fill="#ccc">Fusion + PE</text>
      <text x="400" y="100" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#666">504 tokens</text>
      <text x="400" y="111" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">0 learned params</text>

      <!-- ── Arrow: Fusion → Transformer ───────────────── -->
      <line x1="458" y1="92" x2="484" y2="92" stroke="#444" stroke-width="1.2" marker-end="url(#arr2)"/>
      <text x="471" y="86" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#444">[B,504,128]</text>

      <!-- ── Transformer ×4 (static) ───────────────────── -->
      <rect x="487" y="69" width="132" height="46" rx="5" fill="#0e0e0e" stroke="#555" stroke-width="1.5"/>
      <text x="553" y="87" text-anchor="middle" style="font-size:9px;font-family:monospace" fill="#ccc">Transformer  ×4</text>
      <text x="553" y="100" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#666">MHA + FFN · d=128</text>
      <text x="553" y="111" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">788,992 params</text>

      <!-- ── Arrow: Transformer → Head ─────────────────── -->
      <line x1="619" y1="92" x2="647" y2="92" stroke="#444" stroke-width="1.2" marker-end="url(#arr2)"/>
      <text x="633" y="86" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#444">[B,504,128]</text>

      <!-- ── Output Head (static) ───────────────────────── -->
      <rect x="650" y="69" width="108" height="46" rx="5" fill="#0e0e0e" stroke="#555" stroke-width="1.5"/>
      <text x="704" y="87" text-anchor="middle" style="font-size:9px;font-family:monospace" fill="#ccc">Output Head</text>
      <text x="704" y="100" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#666">MeanPool + Linear</text>
      <text x="704" y="111" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#3a3a3a">258 params</text>

      <!-- ── Arrow: Head → Output ───────────────────────── -->
      <line x1="758" y1="92" x2="784" y2="92" stroke="#444" stroke-width="1.2" marker-end="url(#arr2)"/>
      <text x="771" y="86" text-anchor="middle" style="font-size:7px;font-family:monospace" fill="#444">[B,2]</text>

      <!-- ── Static output node ─────────────────────────── -->
      <rect x="787" y="69" width="128" height="46" rx="5" fill="#0e0e0e" stroke="#3a3a3a" stroke-width="1"/>
      <text x="851" y="87" text-anchor="middle" style="font-size:10px;font-family:monospace" fill="#c0524a">SBP / DBP</text>
      <text x="851" y="101" text-anchor="middle" style="font-size:8px;font-family:monospace" fill="#555">[B, 2]  ·  mmHg</text>
    </svg>

    <!-- Callout — transitions between 3 states -->
    <Transition name="callout" mode="out-in">
      <div v-if="$clicks === 0" key="idle"
           style="border-left:3px solid #333;background:rgba(255,255,255,0.02);border-radius:6px;padding:8px 14px">
        <p style="font-family:monospace;font-size:8.5px;color:#555;line-height:1.7;margin:0">
          The CNN downsampler compresses 15,000 samples → 500 tokens using strided convolution. It makes attention tractable — but at a cost.
        </p>
      </div>
      <div v-else-if="$clicks === 1" key="highlight"
           style="border-left:3px solid #f0c060;background:rgba(240,192,96,0.05);border-radius:6px;padding:8px 14px">
        <p style="font-family:monospace;font-size:8.5px;color:#c8a840;line-height:1.7;margin:0">
          Features in the ~50–100ms range get averaged out by the stride. The dicrotic notch sits in exactly this range — its timing encodes vascular stiffness and is physiologically linked to DBP. This is likely why we missed the diastolic benchmark.
        </p>
      </div>
      <div v-else key="papagei"
           style="border-left:3px solid #c084fc;background:rgba(192,132,252,0.05);border-radius:6px;padding:8px 14px">
        <p style="font-family:monospace;font-size:8.5px;color:#aaa;line-height:1.7;margin:0">
          PapaGei is a PPG foundation model pretrained on large-scale waveform data. Its embeddings capture the full morphological signal — including dicrotic notch timing — without lossy strided convolution. The rest of the pipeline is unchanged.
        </p>
      </div>
    </Transition>

  </div>
</template>

<style scoped>
.callout-enter-active,
.callout-leave-active {
  transition: opacity 0.25s ease, transform 0.25s ease;
}
.callout-enter-from {
  opacity: 0;
  transform: translateY(5px);
}
.callout-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
