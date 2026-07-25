export interface ColorDefinition {
  hex: string;
  bgClass: string;
  label: string;
}

export const STRUCTURE_COLORS: Record<'H' | 'E' | 'C', ColorDefinition> = {
  H: {
    hex: '#3B82F6', // Royal Blue - High contrast, colorblind safe for Helix
    bgClass: 'bg-blue-500/20 border-blue-500/40 text-blue-400',
    label: 'Helix (H)',
  },
  E: {
    hex: '#EA580C', // Warm Orange - High contrast, colorblind safe for Sheet
    bgClass: 'bg-orange-500/20 border-orange-500/40 text-orange-400',
    label: 'Beta Sheet (E)',
  },
  C: {
    hex: '#64748B', // Slate Grey - Neutral background for unstructured loops
    bgClass: 'bg-slate-700/20 border-slate-700/40 text-slate-400',
    label: 'Coil (C)',
  },
};

export const NEUTRAL_COLORS = {
  loadingHex: '#818CF8',   // Indigo - used for flat loading pose and initial sweep state
  lowConfHex: '#334155',    // Uncertain Charcoal - desaturation target for low confidence
  highlightHex: '#A855F7',  // Attention Violet - connection highlight for XAI/Heatmap hover
};
