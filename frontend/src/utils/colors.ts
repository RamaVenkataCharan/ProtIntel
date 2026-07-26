export interface ColorDefinition {
  hex: string;
  bgClass: string;
  label: string;
}

// Q3 Aurora Color Palette
export const STRUCTURE_COLORS: Record<'H' | 'E' | 'C', ColorDefinition> = {
  H: {
    hex: '#7B2FF7', // Vibrant Violet Alpha Helix
    bgClass: 'bg-[#7B2FF7]/20 border-[#7B2FF7]/40 text-[#A16AE8]',
    label: 'Helix (H)',
  },
  E: {
    hex: '#00D9C0', // Cyan/Teal Beta Sheet
    bgClass: 'bg-[#00D9C0]/20 border-[#00D9C0]/40 text-[#66E8D5]',
    label: 'Beta Sheet (E)',
  },
  C: {
    hex: '#E8E8E8', // Light Pearl Coil
    bgClass: 'bg-slate-300/20 border-slate-300/40 text-slate-200',
    label: 'Coil (C)',
  },
};

// Q8 Aurora Color Palette
export const Q8_STRUCTURE_COLORS: Record<'H' | 'G' | 'I' | 'E' | 'B' | 'T' | 'S' | 'C', ColorDefinition> = {
  H: { hex: '#7B2FF7', bgClass: 'bg-[#7B2FF7]/20 border-[#7B2FF7]/40 text-[#A16AE8]', label: 'α-Helix (H)' },
  G: { hex: '#A16AE8', bgClass: 'bg-[#A16AE8]/20 border-[#A16AE8]/40 text-[#C9A6F5]', label: '3₁₀-Helix (G)' },
  I: { hex: '#C9A6F5', bgClass: 'bg-[#C9A6F5]/10 border-[#C9A6F5]/20 text-[#C9A6F5]/60 opacity-60', label: 'π-Helix (I)' },
  E: { hex: '#00D9C0', bgClass: 'bg-[#00D9C0]/20 border-[#00D9C0]/40 text-[#66E8D5]', label: 'β-Sheet (E)' },
  B: { hex: '#66E8D5', bgClass: 'bg-[#66E8D5]/10 border-[#66E8D5]/20 text-[#66E8D5]/60 opacity-60', label: 'β-Bridge (B)' },
  T: { hex: '#FFB347', bgClass: 'bg-[#FFB347]/20 border-[#FFB347]/40 text-[#FFD97D]', label: 'Turn (T)' },
  S: { hex: '#FFD97D', bgClass: 'bg-[#FFD97D]/10 border-[#FFD97D]/20 text-[#FFD97D]/60 opacity-60', label: 'Bend (S)' },
  C: { hex: '#E8E8E8', bgClass: 'bg-slate-300/20 border-slate-300/40 text-slate-200', label: 'Coil (C)' },
};

export const NEUTRAL_COLORS = {
  loadingHex: '#A16AE8',   // Aurora Soft Purple - used for flat loading pose
  lowConfHex: '#334155',    // Uncertain Charcoal - desaturation target for low confidence
  highlightHex: '#00D9C0',  // Aurora Cyan - connection highlight for XAI/Heatmap hover
};

