import React, { useRef, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { generateProteinPath } from '../utils/pathGenerator';

interface ProteinRibbonProps {
  sequence: string;
  q3Prediction: string[];
  confidence: number[];
  hoveredIndex: number | null;
  resetTrigger: number;
  reducedMotion: boolean;
  isMobile: boolean;
}

export const ProteinRibbon: React.FC<ProteinRibbonProps> = ({
  sequence,
  q3Prediction,
  confidence,
  hoveredIndex,
  resetTrigger,
  reducedMotion,
  isMobile,
}) => {
  const instancedMeshRef = useRef<THREE.InstancedMesh>(null);
  
  // Stored path data
  const pathDataRef = useRef(generateProteinPath(sequence, q3Prediction));
  
  // Animation refs
  const revealProgressRef = useRef(reducedMotion ? 1 : 0);
  const pulseIndexRef = useRef<number | null>(null);

  // Re-generate path when sequence or prediction changes
  useEffect(() => {
    pathDataRef.current = generateProteinPath(sequence, q3Prediction);
    revealProgressRef.current = reducedMotion ? 1 : 0;
  }, [sequence, q3Prediction, reducedMotion]);

  // Handle manual reset triggers (e.g. Play Fold-in Again)
  useEffect(() => {
    if (!reducedMotion) {
      revealProgressRef.current = 0;
    }
  }, [resetTrigger, reducedMotion]);

  // Color Palette Definitions
  const colorHelix = new THREE.Color('#E76F51');  // Rust / Coral
  const colorSheet = new THREE.Color('#2A9D8F');  // Teal
  const colorCoil = new THREE.Color('#64748B');   // Charcoal Slate
  const colorPulse = new THREE.Color('#A855F7');  // Attention Violet
  const colorLowConf = new THREE.Color('#334155'); // Grey/slate base for desaturation

  // Temp objects for instance modifications (pre-allocated for performance)
  const tempMatrix = new THREE.Matrix4();
  const tempPosition = new THREE.Vector3();
  const tempDirection = new THREE.Vector3();
  const tempRotation = new THREE.Quaternion();
  const tempScale = new THREE.Vector3();
  const tempColor = new THREE.Color();
  const zAxis = new THREE.Vector3(0, 0, 1);

  useFrame((state, delta) => {
    const mesh = instancedMeshRef.current;
    if (!mesh) return;

    const L = sequence.length;
    const { foldedPoints, unfoldedPoints } = pathDataRef.current;
    const time = state.clock.getElapsedTime();

    // 1. Progress the fold-in reveal animation
    if (revealProgressRef.current < 1 && !reducedMotion) {
      // Slower reveal on larger structures
      const speed = L > 300 ? 0.5 : 0.8;
      revealProgressRef.current = Math.min(1, revealProgressRef.current + delta * speed);
    }

    // 2. Animate the traveling attention pulse position
    if (hoveredIndex !== null) {
      if (pulseIndexRef.current === null) {
        pulseIndexRef.current = 0; // Start pulse from N-terminus
      } else {
        // Sweep towards target hovered index smoothly
        const pulseSpeed = L > 300 ? 50 : 35; // residues per second
        const diff = hoveredIndex - pulseIndexRef.current;
        if (Math.abs(diff) < 0.1) {
          pulseIndexRef.current = hoveredIndex;
        } else {
          pulseIndexRef.current += Math.sign(diff) * Math.min(Math.abs(diff), delta * pulseSpeed);
        }
      }
    } else {
      pulseIndexRef.current = null;
    }

    // 3. Update instances
    const limit = L - 1;
    for (let i = 0; i < limit; i++) {
      const q3 = q3Prediction[i] || 'C';
      const conf = confidence[i] !== undefined ? confidence[i] : 1.0;
      
      // Determine stagger delay for reveal (from N-terminus to C-terminus)
      const staggerDelay = (i / limit) * 0.45; // Max 45% delay offset
      let localReveal = 1;
      
      if (!reducedMotion) {
        // Calculate dynamic reveal progress for this specific residue
        const localVal = (revealProgressRef.current - staggerDelay) / (1 - 0.45);
        localReveal = Math.max(0, Math.min(1, localVal));
      }

      // Smooth coordinate interpolation (flat line -> folded shape)
      const p1_flat = unfoldedPoints[i];
      const p2_flat = unfoldedPoints[i + 1];
      const p1_folded = foldedPoints[i];
      const p2_folded = foldedPoints[i + 1];

      // Interpolated endpoints
      const curP1 = new THREE.Vector3().lerpVectors(p1_flat, p1_folded, localReveal);
      const curP2 = new THREE.Vector3().lerpVectors(p2_flat, p2_folded, localReveal);

      // Center position
      tempPosition.addVectors(curP1, curP2).multiplyScalar(0.5);

      // Length and direction of interpolated segment
      tempDirection.subVectors(curP2, curP1);
      const curLength = tempDirection.length();
      if (curLength > 0.001) {
        tempDirection.normalize();
        tempRotation.setFromUnitVectors(zAxis, tempDirection);
      } else {
        tempRotation.set(0, 0, 0, 1);
      }

      // Define default dimensions based on Secondary Structure class
      let baseWidth = 0.4;
      let baseHeight = 0.4;

      if (q3 === 'H') {
        // Helices are thicker cylindrical coils
        baseWidth = isMobile ? 0.75 : 0.85;
        baseHeight = isMobile ? 0.75 : 0.85;
      } else if (q3 === 'E') {
        // Sheets are flat ribbons
        baseWidth = isMobile ? 1.4 : 1.6;
        baseHeight = isMobile ? 0.12 : 0.16;
      } else {
        // Coils are thin cylinders
        baseWidth = 0.3;
        baseHeight = 0.3;
      }

      // Hover / Pulse scaling effect
      let scaleMult = 1.0;
      let isHighlighted = false;
      
      // Calculate closeness to the traveling attention pulse
      if (pulseIndexRef.current !== null) {
        const distToPulse = Math.abs(i - pulseIndexRef.current);
        if (distToPulse < 3.0) {
          // Gaussian-like pulse peak
          const pulseIntensity = Math.exp(-Math.pow(distToPulse, 2) / 2);
          scaleMult += pulseIntensity * 0.4;
          isHighlighted = true;
        }
      }

      // Direct hover state gets priority scale increase
      if (hoveredIndex !== null && hoveredIndex === i) {
        scaleMult = 1.4;
        isHighlighted = true;
      }

      tempScale.set(baseWidth * scaleMult, baseHeight * scaleMult, curLength || 0.01);

      // Apply transform matrix to instance
      tempMatrix.compose(tempPosition, tempRotation, tempScale);
      mesh.setMatrixAt(i, tempMatrix);

      // 4. Color & Material calculation
      let baseColor = colorCoil;
      if (q3 === 'H') baseColor = colorHelix;
      else if (q3 === 'E') baseColor = colorSheet;

      // Copy base color to work color
      tempColor.copy(baseColor);

      // Confidence desaturation and pulsing logic
      if (conf < 0.70) {
        // Linear interpolation towards gray based on low-confidence severity
        const desatFactor = (0.70 - conf) * 1.3; // max out around 0.5 conf
        tempColor.lerp(colorLowConf, Math.min(0.85, desatFactor));

        // Subtle slow breathe pulse for low-confidence areas to denote uncertainty
        if (!reducedMotion) {
          const pulseIntensity = Math.sin(time * 3 + i * 0.1) * 0.15 + 0.85;
          tempColor.multiplyScalar(pulseIntensity);
        }
      }

      // Blend attention highlight color
      if (isHighlighted && pulseIndexRef.current !== null) {
        const distToPulse = Math.abs(i - pulseIndexRef.current);
        const hoverWeight = hoveredIndex === i ? 1.0 : Math.exp(-Math.pow(distToPulse, 2) / 2);
        tempColor.lerp(colorPulse, hoverWeight * 0.95);
      } else if (hoveredIndex === i) {
        tempColor.lerp(colorPulse, 0.95);
      }

      mesh.setColorAt(i, tempColor);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true;
    }
  });

  return (
    <instancedMesh
      ref={instancedMeshRef}
      args={[null as any, null as any, sequence.length - 1]}
      castShadow
      receiveShadow
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        roughness={0.25}
        metalness={0.6}
        flatShading={false}
      />
    </instancedMesh>
  );
};
