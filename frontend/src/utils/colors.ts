export interface ColorDefinition {
  hex: string;
  bgClass: string;
  label: string;
  mutedHex?: string;
}

// Q3 Aurora/Scientific Color Palette (Vibrant Violet #8B5CF6, Luminous Teal #00E5CC, Crisp Slate #94A3B8)
export const STRUCTURE_COLORS: Record<'H' | 'E' | 'C', ColorDefinition> = {
  H: {
    hex: '#8B5CF6', // Vibrant Violet Alpha Helix (high-contrast)
    mutedHex: 'rgba(139, 92, 246, 0.45)',
    bgClass: 'bg-[#8B5CF6]/20 border-[#8B5CF6]/40 text-[#C4B5FD]',
    label: 'Helix (H)',
  },
  E: {
    hex: '#00E5CC', // Luminous Cyan/Teal Beta Sheet
    mutedHex: 'rgba(0, 229, 204, 0.45)',
    bgClass: 'bg-[#00E5CC]/20 border-[#00E5CC]/40 text-[#5EEAD4]',
    label: 'Beta Sheet (E)',
  },
  C: {
    hex: '#94A3B8', // Technical Slate Coil
    mutedHex: 'rgba(148, 163, 184, 0.35)',
    bgClass: 'bg-slate-700/20 border-slate-600/40 text-slate-300',
    label: 'Coil (C)',
  },
};

// Q8 Scientific Color Palette
export const Q8_STRUCTURE_COLORS: Record<'H' | 'G' | 'I' | 'E' | 'B' | 'T' | 'S' | 'C', ColorDefinition> = {
  H: { hex: '#8B5CF6', mutedHex: 'rgba(139, 92, 246, 0.4)', bgClass: 'bg-[#8B5CF6]/20 border-[#8B5CF6]/40 text-[#C4B5FD]', label: 'α-Helix (H)' },
  G: { hex: '#A855F7', mutedHex: 'rgba(168, 85, 247, 0.4)', bgClass: 'bg-[#A855F7]/20 border-[#A855F7]/40 text-[#D8B4FE]', label: '3₁₀-Helix (G)' },
  I: { hex: '#C084FC', mutedHex: 'rgba(192, 132, 252, 0.3)', bgClass: 'bg-[#C084FC]/10 border-[#C084FC]/20 text-[#E9D5FF]', label: 'π-Helix (I)' },
  E: { hex: '#00E5CC', mutedHex: 'rgba(0, 229, 204, 0.4)', bgClass: 'bg-[#00E5CC]/20 border-[#00E5CC]/40 text-[#5EEAD4]', label: 'β-Sheet (E)' },
  B: { hex: '#2DD4BF', mutedHex: 'rgba(45, 212, 191, 0.3)', bgClass: 'bg-[#2DD4BF]/10 border-[#2DD4BF]/20 text-[#99F6E4]', label: 'β-Bridge (B)' },
  T: { hex: '#F59E0B', mutedHex: 'rgba(245, 158, 11, 0.4)', bgClass: 'bg-[#F59E0B]/20 border-[#F59E0B]/40 text-[#FDE68A]', label: 'Turn (T)' },
  S: { hex: '#FBBF24', mutedHex: 'rgba(251, 191, 36, 0.3)', bgClass: 'bg-[#FBBF24]/10 border-[#FBBF24]/20 text-[#FEF08A]', label: 'Bend (S)' },
  C: { hex: '#94A3B8', mutedHex: 'rgba(148, 163, 184, 0.3)', bgClass: 'bg-slate-700/20 border-slate-600/40 text-slate-300', label: 'Coil (C)' },
};

export const NEUTRAL_COLORS = {
  loadingHex: '#A855F7',     // Soft Violet
  lowConfHex: '#334155',      // Uncertain Slate
  highlightHex: '#00E5CC',    // Aurora Cyan
  amberHighlightHex: '#F59E0B' // Scientific Amber Attention Highlight
};

/**
 * Maps a normalized attribution score (0.0 to 1.0) to a high-contrast heatmap hex string.
 * Cool Blue (#3B82F6) -> Cyan (#06B6D4) -> Amber (#F59E0B) -> Crimson Red (#EF4444)
 */
export function getXAIColorHex(norm: number): string {
  const clamped = Math.max(0, Math.min(1, norm));
  if (clamped < 0.33) {
    // 0.00 -> 0.33: Blue (#3B82F6) to Cyan (#06B6D4)
    const t = clamped / 0.33;
    const r = Math.round(59 + (6 - 59) * t);
    const g = Math.round(130 + (182 - 130) * t);
    const b = Math.round(246 + (212 - 246) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else if (clamped < 0.66) {
    // 0.33 -> 0.66: Cyan (#06B6D4) to Amber (#F59E0B)
    const t = (clamped - 0.33) / 0.33;
    const r = Math.round(6 + (245 - 6) * t);
    const g = Math.round(182 + (158 - 182) * t);
    const b = Math.round(212 + (11 - 212) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // 0.66 -> 1.00: Amber (#F59E0B) to Crimson Red (#EF4444)
    const t = (clamped - 0.66) / 0.34;
    const r = Math.round(245 + (239 - 245) * t);
    const g = Math.round(158 + (68 - 158) * t);
    const b = Math.round(11 + (68 - 11) * t);
    return `rgb(${r}, ${g}, ${b})`;
  }
}
