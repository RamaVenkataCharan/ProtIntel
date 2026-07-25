import * as THREE from 'three';

export interface ProteinPathData {
  foldedPoints: THREE.Vector3[];
  unfoldedPoints: THREE.Vector3[];
  segmentDirections: THREE.Vector3[];
  segmentLengths: number[];
}

/**
 * Procedurally generates 3D path coordinates for a protein sequence.
 * Helices form spirals, sheets form extended zig-zags, and coils form relaxed turns.
 */
export function generateProteinPath(
  sequence: string,
  q3Prediction: string[]
): ProteinPathData {
  const L = sequence.length;
  const foldedPoints: THREE.Vector3[] = [];
  const unfoldedPoints: THREE.Vector3[] = [];

  // 1. Generate flat unfolded points along the X-axis (centered at 0)
  const segmentSpacing = 1.5;
  const totalLength = (L - 1) * segmentSpacing;
  const startX = -totalLength / 2;

  for (let i = 0; i < L; i++) {
    unfoldedPoints.push(new THREE.Vector3(startX + i * segmentSpacing, 0, 0));
  }

  // 2. Generate folded points using a running coordinate frame (Frenet-like frame)
  let pos = new THREE.Vector3(0, 0, 0);
  let forward = new THREE.Vector3(0, 0, 1);
  let up = new THREE.Vector3(0, 1, 0);
  let right = new THREE.Vector3(1, 0, 0);

  foldedPoints.push(pos.clone());

  // We keep track of helical angle to sustain continuous rotation across contiguous helix residues
  let helixAngle = 0;
  // We keep track of sheet step to alternate zig-zag heights
  let sheetStep = 0;

  for (let i = 0; i < L - 1; i++) {
    const q3 = q3Prediction[i] || 'C';
    const nextQ3 = q3Prediction[i + 1] || 'C';

    let stepLength = 1.5;

    if (q3 === 'H') {
      // Helix: Generate a clean cylindrical spiral.
      // Pitch rise per residue ~ 0.8 units, radius ~ 1.2 units, 3.6 residues per turn (100 degrees / 1.745 rad per step)
      stepLength = 0.8;
      helixAngle += 1.745; // 100 degrees in radians

      // Compute local spiral offset
      const spiralOffset = new THREE.Vector3(
        Math.cos(helixAngle) * 1.2,
        Math.sin(helixAngle) * 1.2,
        stepLength
      );

      // Rotate local spiral offset to align with current forward axis
      const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), forward);
      spiralOffset.applyQuaternion(quat);

      pos.add(spiralOffset);

      // Slightly perturb forward direction to make the helix slowly curve in space
      const turnAxis = up.clone().applyAxisAngle(forward, 0.2);
      forward.applyAxisAngle(turnAxis, 0.05).normalize();
      right.crossVectors(forward, up).normalize();
      up.crossVectors(right, forward).normalize();

    } else if (q3 === 'E') {
      // Beta Sheet: extended conformation with zig-zag.
      stepLength = 1.7;
      sheetStep++;

      // Alternating up/down offset for beta strand pleating
      const pleatHeight = 0.25;
      const pleatOffset = up.clone().multiplyScalar((sheetStep % 2 === 0 ? 1 : -1) * pleatHeight);
      
      const stepVec = forward.clone().multiplyScalar(stepLength).add(pleatOffset);
      pos.add(stepVec);

      // Keep beta strands mostly straight
      if (nextQ3 !== 'E') {
        sheetStep = 0; // reset
      }
    } else {
      // Coil: relaxed turns and loops.
      stepLength = 1.5;

      // Add gentle curvatures in coils to fold the protein back on itself (globular structure)
      const turnDirection = Math.sin(i * 0.4) > 0 ? 1 : -1;
      const pitchDirection = Math.cos(i * 0.35) > 0 ? 1 : -1;

      // Rotate local frame
      forward.applyAxisAngle(up, 0.25 * turnDirection);
      forward.applyAxisAngle(right, 0.18 * pitchDirection);
      forward.normalize();

      right.crossVectors(forward, up).normalize();
      up.crossVectors(right, forward).normalize();

      const stepVec = forward.clone().multiplyScalar(stepLength);
      pos.add(stepVec);
    }

    foldedPoints.push(pos.clone());
  }

  // 3. Compute directions and lengths of segments
  const segmentDirections: THREE.Vector3[] = [];
  const segmentLengths: number[] = [];

  for (let i = 0; i < L - 1; i++) {
    const p1 = foldedPoints[i];
    const p2 = foldedPoints[i + 1];
    const dir = new THREE.Vector3().subVectors(p2, p1);
    const len = dir.length();
    segmentLengths.push(len);
    segmentDirections.push(dir.normalize());
  }

  // Center the folded points around the origin (0,0,0) for balanced rotation and viewport fit
  const centroid = new THREE.Vector3(0, 0, 0);
  foldedPoints.forEach((p) => centroid.add(p));
  centroid.divideScalar(L);
  foldedPoints.forEach((p) => p.sub(centroid));

  return {
    foldedPoints,
    unfoldedPoints,
    segmentDirections,
    segmentLengths,
  };
}
