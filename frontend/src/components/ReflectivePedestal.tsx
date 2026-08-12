import React from 'react';
import { MeshReflectorMaterial } from '@react-three/drei';

/**
 * ReflectivePedestal — Subtle glass-plane reflection beneath the protein structure.
 *
 * Provides a "product showcase" feel with very low reflectivity
 * to avoid competing with the protein model visibility.
 *
 * - Positioned at the same Y as the previous ground contact shadow disc
 * - Uses drei's MeshReflectorMaterial for efficient reflection
 * - Skipped on mobile (falls back to the simple shadow disc)
 */

interface ReflectivePedestalProps {
  boundingRadius: number;
  isMobile: boolean;
}

export const ReflectivePedestal: React.FC<ReflectivePedestalProps> = ({
  boundingRadius,
  isMobile,
}) => {
  const size = boundingRadius * 2.5;

  if (isMobile) {
    // Fallback: simple shadow disc (identical to previous implementation)
    return (
      <mesh
        position={[0, -boundingRadius - 2, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
      >
        <circleGeometry args={[boundingRadius * 2.2, 32]} />
        <meshBasicMaterial color="#020408" transparent opacity={0.55} />
      </mesh>
    );
  }

  return (
    <mesh
      position={[0, -boundingRadius - 2, 0]}
      rotation={[-Math.PI / 2, 0, 0]}
      receiveShadow
    >
      <planeGeometry args={[size, size]} />
      <MeshReflectorMaterial
        mirror={0.12}
        blur={[400, 100]}
        color="#040810"
        mixBlur={1}
        mixStrength={0.4}
        roughness={0.88}
        depthScale={0.4}
        minDepthThreshold={0.4}
        maxDepthThreshold={1.2}
        metalness={0.1}
        resolution={512}
      />
    </mesh>
  );
};

export default ReflectivePedestal;
