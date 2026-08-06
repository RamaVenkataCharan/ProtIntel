export interface ColorDefinition {
  hex: string;
  bgClass: string;
  label: string;
  mutedHex?: string;
}

// Q3 Aurora/Scientific Color Palette (Violet #7B2FF7, Teal #00D9C0, Pearl/Muted #475569)
export const STRUCTURE_COLORS: Record<'H' | 'E' | 'C', ColorDefinition> = {
  H: {
    hex: '#7B2FF7', // Vibrant Violet Alpha Helix
    mutedHex: 'rgba(123, 47, 247, 0.4)',
    bgClass: 'bg-[#7B2FF7]/20 border-[#7B2FF7]/40 text-[#A16AE8]',
    label: 'Helix (H)',
  },
  E: {
    hex: '#00D9C0', // Cyan/Teal Beta Sheet
    mutedHex: 'rgba(0, 217, 192, 0.4)',
    bgClass: 'bg-[#00D9C0]/20 border-[#00D9C0]/40 text-[#66E8D5]',
    label: 'Beta Sheet (E)',
  },
  C: {
    hex: '#64748B', // Technical Slate Coil
    mutedHex: 'rgba(100, 116, 139, 0.3)',
    bgClass: 'bg-slate-700/20 border-slate-600/40 text-slate-300',
    label: 'Coil (C)',
  },
};

// Q8 Scientific Color Palette
export const Q8_STRUCTURE_COLORS: Record<'H' | 'G' | 'I' | 'E' | 'B' | 'T' | 'S' | 'C', ColorDefinition> = {
  H: { hex: '#7B2FF7', mutedHex: 'rgba(123, 47, 247, 0.35)', bgClass: 'bg-[#7B2FF7]/20 border-[#7B2FF7]/40 text-[#A16AE8]', label: 'α-Helix (H)' },
  G: { hex: '#9B59F5', mutedHex: 'rgba(155, 89, 245, 0.35)', bgClass: 'bg-[#9B59F5]/20 border-[#9B59F5]/40 text-[#C9A6F5]', label: '3₁₀-Helix (G)' },
  I: { hex: '#C9A6F5', mutedHex: 'rgba(201, 166, 245, 0.25)', bgClass: 'bg-[#C9A6F5]/10 border-[#C9A6F5]/20 text-[#C9A6F5]/60 opacity-60', label: 'π-Helix (I)' },
  E: { hex: '#00D9C0', mutedHex: 'rgba(0, 217, 192, 0.35)', bgClass: 'bg-[#00D9C0]/20 border-[#00D9C0]/40 text-[#66E8D5]', label: 'β-Sheet (E)' },
  B: { hex: '#66E8D5', mutedHex: 'rgba(102, 232, 213, 0.25)', bgClass: 'bg-[#66E8D5]/10 border-[#66E8D5]/20 text-[#66E8D5]/60 opacity-60', label: 'β-Bridge (B)' },
  T: { hex: '#FFB347', mutedHex: 'rgba(255, 179, 71, 0.35)', bgClass: 'bg-[#FFB347]/20 border-[#FFB347]/40 text-[#FFD97D]', label: 'Turn (T)' },
  S: { hex: '#FFD97D', mutedHex: 'rgba(255, 217, 125, 0.25)', bgClass: 'bg-[#FFD97D]/10 border-[#FFD97D]/20 text-[#FFD97D]/60 opacity-60', label: 'Bend (S)' },
  C: { hex: '#64748B', mutedHex: 'rgba(100, 116, 139, 0.25)', bgClass: 'bg-slate-700/20 border-slate-600/40 text-slate-300', label: 'Coil (C)' },
};

export const NEUTRAL_COLORS = {
  loadingHex: '#9B59F5',     // Soft Violet
  lowConfHex: '#334155',      // Uncertain Slate
  highlightHex: '#00D9C0',    // Aurora Cyan
  amberHighlightHex: '#FFB347' // Scientific Amber Attention Highlight
};
