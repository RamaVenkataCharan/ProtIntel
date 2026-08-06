import React, { useRef, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { generateProteinPath } from '../utils/pathGenerator';
import { STRUCTURE_COLORS, Q8_STRUCTURE_COLORS, NEUTRAL_COLORS } from '../utils/colors';

export type ViewMode3D = 'structure' | 'xai' | 'q3' | 'q8';

interface ProteinRibbonProps {
  sequence: string;
  q3Prediction: string[];
  q8Prediction?: string[] | null;
  confidence: number[];
  residueImportance?: number[] | null;
  hoveredIndex: number | null;
  selectedIndex: number | null;
  scrubberIndex: number | null;
  measurePoints: [number, number] | null;
  viewMode: ViewMode3D;
  colorMorph: number; // 0 = Q3, 1 = Q8
  resetTrigger: number;
  reducedMotion: boolean;
  isMobile: boolean;
  onSelectResidue: (index: number | null) => void;
  onHoverResidue: (index: number | null) => void;
}

// XAI Heatmap Colormap Generator (Cool Blue -> Amber -> Crimson Red)
const getHeatmapColor = (importance: number): THREE.Color => {
  const norm = Math.max(0, Math.min(1, importance));
  const coolBlue = new THREE.Color('#3B82F6');
  const warmAmber = new THREE.Color('#F59E0B');
  const hotCrimson = new THREE.Color('#EF4444');
  
  const col = new THREE.Color();
  if (norm < 0.5) {
    col.lerpColors(coolBlue, warmAmber, norm * 2);
  } else {
    col.lerpColors(warmAmber, hotCrimson, (norm - 0.5) * 2);
  }
  return col;
};

export const ProteinRibbon: React.FC<ProteinRibbonProps> = ({
  sequence,
  q3Prediction,
  q8Prediction,
  confidence,
  residueImportance,
  hoveredIndex,
  selectedIndex,
  scrubberIndex,
  measurePoints,
  viewMode,
  colorMorph,
  resetTrigger,
  reducedMotion,
  isMobile,
  onSelectResidue,
  onHoverResidue,
}) => {
  const instancedMeshRef = useRef<THREE.InstancedMesh>(null);
  
  // Stored path data
  const pathDataRef = useRef(generateProteinPath(sequence, q3Prediction));
  const revealProgressRef = useRef(reducedMotion ? 1 : 0);
  const pulseIndexRef = useRef<number | null>(null);

  // Re-generate path when sequence or prediction changes
  useEffect(() => {
    pathDataRef.current = generateProteinPath(sequence, q3Prediction);
    revealProgressRef.current = reducedMotion ? 1 : 0;
  }, [sequence, q3Prediction, reducedMotion]);

  useEffect(() => {
    if (!reducedMotion) {
      revealProgressRef.current = 0;
    }
  }, [resetTrigger, reducedMotion]);

  // Pre-allocated temporaries for instance loop
  const tempMatrix = new THREE.Matrix4();
  const tempPosition = new THREE.Vector3();
  const tempDirection = new THREE.Vector3();
  const tempRotation = new THREE.Quaternion();
  const tempScale = new THREE.Vector3();
  const tempColor = new THREE.Color();
  const colorNeutral = new THREE.Color(NEUTRAL_COLORS.loadingHex);
  const colorHoverGlow = new THREE.Color('#00D9C0');
  const colorSelectGlow = new THREE.Color('#FFB347');
  const colorLowConf = new THREE.Color('#334155');
  const zAxis = new THREE.Vector3(0, 0, 1);

  useFrame((state, delta) => {
    const mesh = instancedMeshRef.current;
    if (!mesh) return;

    const L = sequence.length;
    const { foldedPoints, unfoldedPoints } = pathDataRef.current;
    const time = state.clock.getElapsedTime();

    // 1. Progress reveal animation
    if (revealProgressRef.current < 1 && !reducedMotion) {
      const speed = L > 300 ? 0.5 : 0.85;
      revealProgressRef.current = Math.min(1, revealProgressRef.current + delta * speed);
    }

    // 2. Pulse position interpolation towards hover/scrubber
    const targetTarget = hoveredIndex !== null ? hoveredIndex : scrubberIndex;
    if (targetTarget !== null) {
      if (pulseIndexRef.current === null) {
        pulseIndexRef.current = targetTarget;
      } else {
        const pulseSpeed = L > 300 ? 50 : 35;
        const diff = targetTarget - pulseIndexRef.current;
        if (Math.abs(diff) < 0.1) {
          pulseIndexRef.current = targetTarget;
        } else {
          pulseIndexRef.current += Math.sign(diff) * Math.min(Math.abs(diff), delta * pulseSpeed);
        }
      }
    } else {
      pulseIndexRef.current = null;
    }

    // Find max importance for normalization if XAI mode
    let maxImportance = 1.0;
    if (residueImportance && residueImportance.length > 0) {
      maxImportance = Math.max(...residueImportance.map(v => Math.abs(v))) || 1.0;
    }

    const limit = L - 1;
    for (let i = 0; i < limit; i++) {
      const q3 = q3Prediction[i] || 'C';
      const conf = confidence[i] !== undefined ? confidence[i] : 1.0;
      const importance = residueImportance?.[i] ?? 0;
      const normImportance = Math.abs(importance) / maxImportance;

      const staggerDelay = (i / limit) * 0.45;
      let localReveal = 1;
      if (!reducedMotion) {
        const localVal = (revealProgressRef.current - staggerDelay) / (1 - 0.45);
        localReveal = Math.max(0, Math.min(1, localVal));
      }

      const p1_flat = unfoldedPoints[i];
      const p2_flat = unfoldedPoints[i + 1];
      const p1_folded = foldedPoints[i];
      const p2_folded = foldedPoints[i + 1];

      const curP1 = new THREE.Vector3().lerpVectors(p1_flat, p1_folded, localReveal);
      const curP2 = new THREE.Vector3().lerpVectors(p2_flat, p2_folded, localReveal);

      tempPosition.addVectors(curP1, curP2).multiplyScalar(0.5);

      tempDirection.subVectors(curP2, curP1);
      const curLength = tempDirection.length();
      if (curLength > 0.001) {
        tempDirection.normalize();
        tempRotation.setFromUnitVectors(zAxis, tempDirection);
      } else {
        tempRotation.set(0, 0, 0, 1);
      }

      // Geometry Dimensions
      let baseWidth = 0.4;
      let baseHeight = 0.4;
      if (q3 === 'H') {
        baseWidth = isMobile ? 0.7 : 0.8;
        baseHeight = isMobile ? 0.7 : 0.8;
      } else if (q3 === 'E') {
        baseWidth = isMobile ? 2.2 : 2.6;
        baseHeight = isMobile ? 0.08 : 0.10;
      } else {
        baseWidth = 0.22;
        baseHeight = 0.22;
      }

      // Scale multipliers for hover / select / scrubber / measurement
      let scaleMult = 1.0;
      const isHov = hoveredIndex === i;
      const isSel = selectedIndex === i;
      const isScrub = scrubberIndex === i;
      const isMeasured = measurePoints ? (measurePoints[0] === i || measurePoints[1] === i) : false;

      if (isSel) scaleMult = 1.6;
      else if (isHov) scaleMult = 1.4;
      else if (isScrub) scaleMult = 1.45;
      else if (isMeasured) scaleMult = 1.5;

      tempScale.set(baseWidth * scaleMult, baseHeight * scaleMult, curLength || 0.01);
      tempMatrix.compose(tempPosition, tempRotation, tempScale);
      mesh.setMatrixAt(i, tempMatrix);

      // Color computation based on View Mode
      const q3Key = (q3 === 'H' || q3 === 'E' || q3 === 'C') ? q3 : 'C';
      const q8Char = q8Prediction ? q8Prediction[i] : null;
      const q8Key = (q8Char && q8Char in Q8_STRUCTURE_COLORS) ? (q8Char as keyof typeof Q8_STRUCTURE_COLORS) : null;

      const colorQ3 = new THREE.Color(STRUCTURE_COLORS[q3Key].hex);
      const colorQ8 = new THREE.Color(q8Key ? Q8_STRUCTURE_COLORS[q8Key].hex : STRUCTURE_COLORS[q3Key].hex);

      // Interpolate between Q3 and Q8 structure colors via colorMorph slider
      const colorStruct = new THREE.Color().lerpColors(colorQ3, colorQ8, colorMorph);

      let targetColor: THREE.Color;

      if (viewMode === 'xai') {
        targetColor = getHeatmapColor(normImportance);
      } else if (viewMode === 'q3') {
        targetColor = colorQ3;
      } else if (viewMode === 'q8') {
        targetColor = colorQ8;
      } else {
        targetColor = colorStruct;
      }

      tempColor.lerpColors(colorNeutral, targetColor, localReveal);

      // Confidence-Based Opacity / Desaturation Encoding
      if (conf < 0.65) {
        const desatFactor = (0.65 - conf) * 1.4;
        tempColor.lerp(colorLowConf, Math.min(0.85, desatFactor));
        if (!reducedMotion) {
          const pulseIntensity = Math.sin(time * 3 + i * 0.1) * 0.15 + 0.85;
          tempColor.multiplyScalar(pulseIntensity);
        }
      }

      // Priority Highlights: Selection (Amber) > Hover / Scrubber (Teal)
      if (isSel || isMeasured) {
        tempColor.lerp(colorSelectGlow, 0.95);
      } else if (isHov || isScrub) {
        tempColor.lerp(colorHoverGlow, 0.92);
      }

      mesh.setColorAt(i, tempColor);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={instancedMeshRef}
      args={[null as any, null as any, sequence.length - 1]}
      castShadow
      receiveShadow
      onClick={(e) => {
        e.stopPropagation();
        if (e.instanceId !== undefined) {
          onSelectResidue(e.instanceId);
        }
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        if (e.instanceId !== undefined) {
          onHoverResidue(e.instanceId);
        }
      }}
      onPointerOut={(e) => {
        e.stopPropagation();
        onHoverResidue(null);
      }}
    >
      <cylinderGeometry args={[0.5, 0.5, 1, 8]} />
      <meshStandardMaterial
        roughness={0.25}
        metalness={0.55}
        emissive="#1a0b2e"
        emissiveIntensity={0.2}
        flatShading={false}
      />
    </instancedMesh>
  );
};
