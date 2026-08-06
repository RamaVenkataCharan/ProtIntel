import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { ProteinRibbon, type ViewMode3D } from './ProteinRibbon';
import { RotateCcw, Play, Pause, Ruler, Camera, X, Download, Keyboard } from 'lucide-react';
import { useThemeStore } from '../store/useThemeStore';
import { generateProteinPath } from '../utils/pathGenerator';

interface ProteinStructure3DProps {
  sequence: string | null;
  q3Prediction: string[] | null;
  q8Prediction?: string[] | null;
  confidence: number[] | null;
  residueImportance?: number[] | null;
  hoveredIndex: number | null;
  isPredicting: boolean;
  onHoverResidue?: (idx: number | null) => void;
  viewMode?: ViewMode3D;
  onViewModeChange?: (mode: ViewMode3D) => void;
}

// CameraController: Manages smooth 3/4 framing, target focus, reset, and cinematic fly-through
interface CameraControllerProps {
  targetPosition: THREE.Vector3 | null;
  isCinematicTour: boolean;
  foldedPoints: THREE.Vector3[];
  reducedMotion: boolean;
  isInteracting: React.MutableRefObject<boolean>;
  orbitControlsRef: React.RefObject<any>;
  defaultCamPosition: THREE.Vector3;
  resetTrigger: number;
  onEndTour: () => void;
}

const CameraController: React.FC<CameraControllerProps> = ({
  targetPosition,
  isCinematicTour,
  foldedPoints,
  reducedMotion,
  isInteracting,
  orbitControlsRef,
  defaultCamPosition,
  resetTrigger,
  onEndTour,
}) => {
  const tourProgressRef = useRef(0);
  const isResettingRef = useRef(false);

  // Sync OrbitControls enabled state when tour starts / ends
  useEffect(() => {
    const controls = orbitControlsRef.current;
    if (!controls) return;

    if (isCinematicTour) {
      tourProgressRef.current = 0;
      controls.enabled = false;
    } else {
      controls.enabled = true;
      controls.update();
    }
  }, [isCinematicTour, orbitControlsRef]);

  // Smoothly lerp to default 3/4 isometric position on resetTrigger or new sequence
  useEffect(() => {
    isResettingRef.current = true;
    const timer = setTimeout(() => {
      isResettingRef.current = false;
    }, 800);
    return () => clearTimeout(timer);
  }, [resetTrigger, defaultCamPosition]);

  useFrame((state, delta) => {
    const controls = orbitControlsRef.current;

    // 1. Resetting to 3/4 dynamic bounding-sphere framing
    if (isResettingRef.current && !isInteracting.current && !isCinematicTour) {
      state.camera.position.lerp(defaultCamPosition, delta * 5);
      if (controls) {
        controls.target.lerp(new THREE.Vector3(0, 0, 0), delta * 5);
        controls.update();
      }
      return;
    }

    // 2. ONE-SHOT CINEMATIC FLY-THROUGH TOUR (Stable along backbone, no coordinate fight)
    if (isCinematicTour && foldedPoints.length > 0 && !reducedMotion && !isInteracting.current) {
      tourProgressRef.current += delta * 0.12; // Smooth cinematic velocity

      if (tourProgressRef.current >= 1.0) {
        tourProgressRef.current = 1.0;
        if (controls) {
          const finalPt = foldedPoints[foldedPoints.length - 1];
          controls.target.copy(finalPt);
          controls.enabled = true;
          controls.update();
        }
        onEndTour();
        return;
      }

      const pointIdx = Math.floor(tourProgressRef.current * (foldedPoints.length - 1));
      const pt = foldedPoints[pointIdx] || foldedPoints[0];

      // Smooth camera offset floating slightly above & right of current residue
      const camPos = new THREE.Vector3(pt.x + 6, pt.y + 4, pt.z + 12);
      state.camera.position.lerp(camPos, delta * 4);
      state.camera.lookAt(pt);

      if (controls) {
        controls.target.copy(pt);
      }
      return;
    }

    // 3. Click-to-Inspect Focus Target
    if (targetPosition && !isInteracting.current) {
      const camPos = new THREE.Vector3(targetPosition.x, targetPosition.y + 4, targetPosition.z + 16);
      state.camera.position.lerp(camPos, delta * 4);
      if (controls) {
        controls.target.lerp(targetPosition, delta * 4);
        controls.update();
      }
    }
  });

  return null;
};

// GroupRig handles slow idle ambient spin (safely paused when inspecting or during tour/playback)
interface GroupRigProps {
  children: React.ReactNode;
  reducedMotion: boolean;
  isInteracting: React.MutableRefObject<boolean>;
  isCinematicTour: boolean;
  isPlaying: boolean;
  hasSelection: boolean;
}

const GroupRig: React.FC<GroupRigProps> = ({
  children,
  reducedMotion,
  isInteracting,
  isCinematicTour,
  isPlaying,
  hasSelection,
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const currentRotation = useRef(0);

  // Pause ambient spin during active user interaction, tour, timeline playback, or residue inspection
  const shouldSpin = !isInteracting.current && !reducedMotion && !isCinematicTour && !isPlaying && !hasSelection;

  useFrame((_, delta) => {
    if (groupRef.current && shouldSpin) {
      currentRotation.current += delta * 0.04;
      groupRef.current.rotation.y = currentRotation.current;
    }
  });

  return <group ref={groupRef}>{children}</group>;
};

export const ProteinStructure3D: React.FC<ProteinStructure3DProps> = ({
  sequence,
  q3Prediction,
  q8Prediction,
  confidence,
  residueImportance,
  hoveredIndex,
  isPredicting,
  onHoverResidue,
  viewMode: controlledViewMode,
  onViewModeChange,
}) => {
  const { theme } = useThemeStore();
  const [reducedMotion, setReducedMotion] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [resetTrigger, setResetTrigger] = useState(0);
  const isInteracting = useRef(false);
  const orbitControlsRef = useRef<any>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);

  // Interactive 3D States
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [internalViewMode, setInternalViewMode] = useState<ViewMode3D>('structure');
  const viewMode = controlledViewMode !== undefined ? controlledViewMode : internalViewMode;

  const handleViewModeChange = (mode: ViewMode3D) => {
    setInternalViewMode(mode);
    onViewModeChange?.(mode);
  };

  const [colorMorph, setColorMorph] = useState(0); // 0 = Q3, 1 = Q8
  
  // Scrubber / Playback
  const [scrubberIndex, setScrubberIndex] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState<1 | 2 | 4>(1);

  // Measurement tool
  const [isMeasuringMode, setIsMeasuringMode] = useState(false);
  const [measurePoints, setMeasurePoints] = useState<[number, number] | null>(null);

  // Cinematic tour
  const [isCinematicTour, setIsCinematicTour] = useState(false);

  // Shortcuts Modal
  const [showShortcuts, setShowShortcuts] = useState(false);

  // Detect reduced motion & mobile
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(mediaQuery.matches);
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  useEffect(() => {
    const checkDevice = () => setIsMobile(window.innerWidth < 768 || navigator.maxTouchPoints > 1);
    checkDevice();
    window.addEventListener('resize', checkDevice);
    return () => window.removeEventListener('resize', checkDevice);
  }, []);

  // Keyboard Shortcuts Listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      
      if (e.key === ' ') {
        e.preventDefault();
        setIsPlaying((p) => !p);
      } else if (e.key === 'Escape') {
        setSelectedIndex(null);
        setMeasurePoints(null);
        setIsCinematicTour(false);
      } else if (e.key.toLowerCase() === 'r') {
        triggerReset();
      } else if (e.key.toLowerCase() === 'm') {
        setIsMeasuringMode((m) => !m);
      } else if (e.key.toLowerCase() === 't') {
        setIsCinematicTour((t) => !t);
      } else if (e.key === '?') {
        setShowShortcuts((s) => !s);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Compute backbone points for click-focus & measurement
  const pathData = useMemo(() => {
    if (!sequence || !q3Prediction) return null;
    return generateProteinPath(sequence, q3Prediction);
  }, [sequence, q3Prediction]);

  // Dynamic Bounding Sphere & Optimal 3/4 Isometric Perspective Camera Calculation (BUG 4 Fix)
  const { boundingRadius, defaultCamDistance, defaultCamPosition } = useMemo(() => {
    if (!pathData?.foldedPoints || pathData.foldedPoints.length === 0) {
      const defaultDist = 25;
      const dir = new THREE.Vector3(0.42, 0.32, 0.85).normalize();
      return {
        boundingRadius: 8,
        defaultCamDistance: defaultDist,
        defaultCamPosition: dir.multiplyScalar(defaultDist),
      };
    }

    let maxDistSq = 0;
    for (const p of pathData.foldedPoints) {
      const dSq = p.x * p.x + p.y * p.y + p.z * p.z;
      if (dSq > maxDistSq) maxDistSq = dSq;
    }
    const radius = Math.max(6, Math.sqrt(maxDistSq));
    
    // Vertical FOV = 40 deg; frame bounding sphere with 35% comfortable padding
    const fovRad = (40 * Math.PI) / 180;
    const requiredDist = (radius * 1.35) / Math.sin(fovRad / 2);
    const dist = Math.max(18, Math.min(130, requiredDist));
    
    // 3/4 isometric perspective direction vector (elevated 3/4 view)
    const dir = new THREE.Vector3(0.42, 0.32, 0.85).normalize();
    const pos = dir.multiplyScalar(dist);

    return {
      boundingRadius: radius,
      defaultCamDistance: dist,
      defaultCamPosition: pos,
    };
  }, [pathData]);

  // Handle residue click inspect
  const handleSelectResidue = (idx: number | null) => {
    if (isCinematicTour) {
      setIsCinematicTour(false);
    }
    if (isMeasuringMode) {
      if (idx === null) return;
      if (!measurePoints) {
        setMeasurePoints([idx, idx]);
      } else {
        setMeasurePoints([measurePoints[0], idx]);
        setIsMeasuringMode(false);
      }
      return;
    }
    setSelectedIndex(idx);
    if (onHoverResidue) onHoverResidue(idx);
  };

  const handleHoverResidue = (idx: number | null) => {
    if (onHoverResidue) onHoverResidue(idx);
  };

  // High-Res Canvas PNG Snapshot Export
  const handleTakeSnapshot = () => {
    const canvas = canvasContainerRef.current?.querySelector('canvas');
    if (!canvas) return;
    try {
      const dataUrl = canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = dataUrl;
      a.download = `ProtIntel_3D_Structure_${Date.now()}.png`;
      a.click();
    } catch (err) {
      console.error('Snapshot failed:', err);
    }
  };

  // Playback Auto-Advance
  useEffect(() => {
    if (!isPlaying || !sequence) return;
    const intervalTime = 300 / playSpeed;
    const timer = setInterval(() => {
      setScrubberIndex((prev) => {
        const next = (prev === null ? 0 : prev + 1);
        if (next >= sequence.length) {
          setIsPlaying(false);
          return 0;
        }
        return next;
      });
    }, intervalTime);
    return () => clearInterval(timer);
  }, [isPlaying, sequence, playSpeed]);

  const triggerReset = () => {
    setResetTrigger((prev) => prev + 1);
    setSelectedIndex(null);
    setScrubberIndex(null);
    setMeasurePoints(null);
    setIsCinematicTour(false);
  };

  const hasData = Boolean(sequence && q3Prediction && confidence);
  const L = sequence?.length || 0;

  // Selected Residue Target Position
  const selectedTargetPosition = useMemo(() => {
    const activeIdx = selectedIndex ?? scrubberIndex;
    if (activeIdx === null || !pathData || !pathData.foldedPoints[activeIdx]) return null;
    return pathData.foldedPoints[activeIdx];
  }, [selectedIndex, scrubberIndex, pathData]);

  // Measured distance calculation
  const measurementDistance = useMemo(() => {
    if (!measurePoints || !pathData) return null;
    const p1 = pathData.foldedPoints[measurePoints[0]];
    const p2 = pathData.foldedPoints[measurePoints[1]];
    if (!p1 || !p2) return null;
    return p1.distanceTo(p2);
  }, [measurePoints, pathData]);

  const activeResidueInfo = useMemo(() => {
    const idx = selectedIndex ?? hoveredIndex ?? scrubberIndex;
    if (idx === null || !sequence || !q3Prediction || !confidence) return null;
    return {
      index: idx,
      aa: sequence[idx],
      q3: q3Prediction[idx],
      q8: q8Prediction?.[idx] || q3Prediction[idx],
      conf: confidence[idx],
      importance: residueImportance?.[idx],
    };
  }, [selectedIndex, hoveredIndex, scrubberIndex, sequence, q3Prediction, q8Prediction, confidence, residueImportance]);

  return (
    <div
      ref={canvasContainerRef}
      className="w-full h-[460px] bg-[radial-gradient(circle_at_50%_30%,#161b2e_0%,#090d18_70%,#04060b_100%)] border border-[var(--border-subtle)] rounded-3xl relative overflow-hidden flex flex-col group/canvas shadow-2xl"
    >
      {/* ── TOP TOOLBAR ─────────────────────────────────────────────── */}
      <div className="absolute top-3 left-3 right-3 z-20 flex flex-wrap items-center justify-between gap-2 pointer-events-auto">
        <div className="flex flex-wrap items-center gap-1.5 bg-black/75 backdrop-blur-md p-1.5 rounded-2xl border border-white/[0.1] shadow-lg">
          <button
            onClick={triggerReset}
            disabled={!hasData || isPredicting}
            className="px-2.5 py-1 rounded-xl text-xs font-bold text-slate-300 hover:text-white hover:bg-white/10 transition-all flex items-center gap-1 cursor-pointer disabled:opacity-30"
            title="Reset to 3/4 perspective (R)"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>Reset View</span>
          </button>

          {/* View Mode Switcher */}
          <div className="h-4 w-[1px] bg-white/10 mx-1" />
          <button
            onClick={() => handleViewModeChange('structure')}
            className={`px-2.5 py-1 rounded-xl text-xs font-bold transition-all cursor-pointer ${
              viewMode === 'structure' ? 'bg-violet-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Structure Mode
          </button>
          <button
            onClick={() => handleViewModeChange('xai')}
            disabled={!residueImportance}
            className={`px-2.5 py-1 rounded-xl text-xs font-bold transition-all cursor-pointer ${
              viewMode === 'xai' ? 'bg-amber-500 text-black shadow-md' : 'text-slate-400 hover:text-slate-200 disabled:opacity-30'
            }`}
          >
            XAI Heatmap
          </button>

          {/* Q3 vs Q8 Morph Slider */}
          <div className="h-4 w-[1px] bg-white/10 mx-1" />
          <span className="text-[10px] font-mono text-slate-400 font-bold px-1">Q3</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={colorMorph}
            onChange={(e) => setColorMorph(parseFloat(e.target.value))}
            className="w-16 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-400"
            title="Morph colors between Q3 and Q8"
          />
          <span className="text-[10px] font-mono text-slate-400 font-bold px-1">Q8</span>
        </div>

        {/* Action Tools: Snapshot, Measurement, Tour, Keyboard Guide */}
        <div className="flex items-center gap-1.5 bg-black/75 backdrop-blur-md p-1.5 rounded-2xl border border-white/[0.1] shadow-lg">
          <button
            onClick={handleTakeSnapshot}
            disabled={!hasData || isPredicting}
            className="px-2 py-1 rounded-xl text-xs font-bold text-slate-300 hover:text-white hover:bg-white/10 transition-all flex items-center gap-1 cursor-pointer disabled:opacity-30"
            title="Take 3D High-Res PNG Snapshot"
          >
            <Download className="h-3.5 w-3.5 text-teal-400" />
            <span>Snapshot</span>
          </button>

          <button
            onClick={() => {
              setIsMeasuringMode(!isMeasuringMode);
              if (!isMeasuringMode) setMeasurePoints(null);
            }}
            className={`px-2 py-1 rounded-xl text-xs font-bold transition-all flex items-center gap-1 cursor-pointer ${
              isMeasuringMode ? 'bg-amber-500 text-black shadow-md' : 'text-slate-300 hover:text-white hover:bg-white/10'
            }`}
            title="Measure schematic distance (M)"
          >
            <Ruler className="h-3.5 w-3.5" />
            <span>Measure</span>
          </button>

          <button
            onClick={() => {
              setIsCinematicTour(!isCinematicTour);
              if (!isCinematicTour) setSelectedIndex(null);
            }}
            className={`px-2 py-1 rounded-xl text-xs font-bold transition-all flex items-center gap-1 cursor-pointer ${
              isCinematicTour ? 'bg-violet-500 text-white shadow-md' : 'text-slate-300 hover:text-white hover:bg-white/10'
            }`}
            title="Cinematic fly-through camera (T)"
          >
            <Camera className="h-3.5 w-3.5" />
            <span>Tour</span>
          </button>

          <button
            onClick={() => setShowShortcuts(!showShortcuts)}
            className="px-2 py-1 rounded-xl text-xs font-bold text-slate-400 hover:text-white hover:bg-white/10 transition-all cursor-pointer"
            title="Keyboard Shortcuts (?)"
          >
            <Keyboard className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* ── KEYBOARD SHORTCUTS MODAL ─────────────────────────────────── */}
      {showShortcuts && (
        <div className="absolute inset-0 z-30 bg-black/80 backdrop-blur-md flex items-center justify-center p-6 animate-fade-in">
          <div className="glass-instrument-card p-6 rounded-3xl max-w-sm w-full border border-violet-500/30 shadow-2xl">
            <div className="flex items-center justify-between mb-4 pb-2 border-b border-white/10">
              <h4 className="text-sm font-bold text-slate-100 flex items-center gap-2" style={{ fontFamily: 'var(--font-heading)' }}>
                <Keyboard className="h-4 w-4 text-violet-400" />
                Keyboard Shortcuts
              </h4>
              <button onClick={() => setShowShortcuts(false)} className="text-slate-400 hover:text-white cursor-pointer">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs font-mono">
              <div><kbd className="px-2 py-0.5 bg-white/10 rounded font-bold text-amber-400">Space</kbd> Play/Pause</div>
              <div><kbd className="px-2 py-0.5 bg-white/10 rounded font-bold text-amber-400">R</kbd> Reset Camera</div>
              <div><kbd className="px-2 py-0.5 bg-white/10 rounded font-bold text-amber-400">Esc</kbd> Clear Selection</div>
              <div><kbd className="px-2 py-0.5 bg-white/10 rounded font-bold text-amber-400">M</kbd> Distance Tool</div>
              <div><kbd className="px-2 py-0.5 bg-white/10 rounded font-bold text-amber-400">T</kbd> Fly-Through Tour</div>
              <div><kbd className="px-2 py-0.5 bg-white/10 rounded font-bold text-amber-400">?</kbd> Shortcuts Guide</div>
            </div>
          </div>
        </div>
      )}

      {/* ── FLOATING HUD RESIDUE INSPECT CARD ───────────────────────── */}
      {activeResidueInfo && (
        <div className="absolute top-16 left-4 z-20 glass-instrument-card p-4 rounded-2xl max-w-xs animate-spring-up border border-teal-400/30 shadow-2xl">
          <div className="flex items-center justify-between mb-2 pb-2 border-b border-white/10">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-teal-400 animate-pulse" />
              <span className="text-xs font-mono font-bold text-teal-300">
                RESIDUE_INSPECT // {activeResidueInfo.aa}{activeResidueInfo.index + 1}
              </span>
            </div>
            <button
              onClick={() => { setSelectedIndex(null); handleHoverResidue(null); }}
              className="text-slate-400 hover:text-white cursor-pointer"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div>
              <span className="text-[9px] text-slate-500 font-bold uppercase block">Position</span>
              <span className="text-slate-200 font-bold">Index {activeResidueInfo.index + 1}</span>
            </div>
            <div>
              <span className="text-[9px] text-slate-500 font-bold uppercase block">Confidence</span>
              <span className="text-teal-400 font-bold">{(activeResidueInfo.conf * 100).toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-[9px] text-slate-500 font-bold uppercase block">Q3 Structure</span>
              <span className="text-violet-400 font-bold">{activeResidueInfo.q3}</span>
            </div>
            <div>
              <span className="text-[9px] text-slate-500 font-bold uppercase block">Q8 DSSP</span>
              <span className="text-amber-400 font-bold">{activeResidueInfo.q8}</span>
            </div>
            {activeResidueInfo.importance !== undefined && (
              <div className="col-span-2 pt-1 border-t border-white/5">
                <span className="text-[9px] text-slate-500 font-bold uppercase block">XAI Attribution Score</span>
                <span className="text-emerald-400 font-bold">{activeResidueInfo.importance.toFixed(4)}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── MEASUREMENT HUD TAG ─────────────────────────────────────── */}
      {measurePoints && measurementDistance !== null && (
        <div className="absolute top-16 right-4 z-20 glass-instrument-card p-3 rounded-2xl animate-spring-up border border-amber-400/40 shadow-2xl">
          <div className="flex items-center justify-between gap-2 mb-1">
            <span className="text-[10px] font-mono font-bold text-amber-400">SCHEMATIC_MEASUREMENT</span>
            <button onClick={() => setMeasurePoints(null)} className="text-slate-400 hover:text-white cursor-pointer">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          <p className="text-sm font-bold font-mono text-white">
            Residues {measurePoints[0] + 1} ↔ {measurePoints[1] + 1}: <span className="text-amber-400">{measurementDistance.toFixed(2)} units</span>
          </p>
          <span className="text-[9px] text-amber-400/80 font-mono block mt-1">
            ⚠️ Illustrative ribbon distance (not physical Ångstroms)
          </span>
        </div>
      )}

      {/* ── THREE.JS CANVAS ─────────────────────────────────────────── */}
      <div className="w-full flex-1">
        <Canvas
          shadows
          gl={{ antialias: !isMobile, powerPreference: 'high-performance', preserveDrawingBuffer: true }}
          camera={{
            fov: 40,
            near: 0.1,
            far: 300,
            position: [defaultCamPosition.x, defaultCamPosition.y, defaultCamPosition.z],
          }}
        >
          <color attach="background" args={[theme === 'dark' ? '#070a12' : '#f1f5f9']} />
          <fog attach="fog" args={[theme === 'dark' ? '#070a12' : '#f1f5f9', 60, 250]} />

          {/* Studio 4-Point Scientific Lighting Rig */}
          <ambientLight intensity={0.9} />
          <directionalLight
            position={[20, 30, 25]}
            intensity={1.8}
            castShadow
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
          />
          <directionalLight position={[-20, 15, 20]} intensity={0.9} color="#E0F2FE" />
          <directionalLight position={[0, -12, -30]} intensity={2.4} color="#C4B5FD" />
          <pointLight position={[0, -20, 0]} intensity={0.8} color="#00E5CC" />

          {/* Dynamic Ground Contact Shadow Disc */}
          <mesh position={[0, -boundingRadius - 2, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
            <circleGeometry args={[boundingRadius * 2.2, 32]} />
            <meshBasicMaterial color="#020408" transparent opacity={0.55} />
          </mesh>

          {/* Dynamic Camera Controller with Smooth Handover */}
          <CameraController
            targetPosition={selectedTargetPosition}
            isCinematicTour={isCinematicTour}
            foldedPoints={pathData?.foldedPoints || []}
            reducedMotion={reducedMotion}
            isInteracting={isInteracting}
            orbitControlsRef={orbitControlsRef}
            defaultCamPosition={defaultCamPosition}
            resetTrigger={resetTrigger}
            onEndTour={() => setIsCinematicTour(false)}
          />

          <OrbitControls
            ref={orbitControlsRef}
            enableDamping
            dampingFactor={0.05}
            maxDistance={defaultCamDistance * 3.2}
            minDistance={Math.max(4, defaultCamDistance * 0.15)}
            onStart={() => {
              isInteracting.current = true;
              if (isCinematicTour) setIsCinematicTour(false);
            }}
            onEnd={() => {
              isInteracting.current = false;
            }}
          />

          <GroupRig
            reducedMotion={reducedMotion}
            isInteracting={isInteracting}
            isCinematicTour={isCinematicTour}
            isPlaying={isPlaying}
            hasSelection={selectedIndex !== null}
          >
            {hasData && (
              <ProteinRibbon
                sequence={sequence!}
                q3Prediction={q3Prediction!}
                q8Prediction={q8Prediction}
                confidence={confidence!}
                residueImportance={residueImportance}
                hoveredIndex={hoveredIndex}
                selectedIndex={selectedIndex}
                scrubberIndex={scrubberIndex}
                measurePoints={measurePoints}
                viewMode={viewMode}
                colorMorph={colorMorph}
                resetTrigger={resetTrigger}
                reducedMotion={reducedMotion}
                isMobile={isMobile}
                onSelectResidue={handleSelectResidue}
                onHoverResidue={handleHoverResidue}
              />
            )}
          </GroupRig>
        </Canvas>
      </div>

      {/* ── TIMELINE SCRUBBER & PLAYBACK CONTROLS ───────────────────── */}
      {hasData && (
        <div className="px-4 py-2.5 bg-black/85 backdrop-blur-md border-t border-white/10 flex items-center justify-between gap-4 z-20 select-none shadow-lg">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-1.5 rounded-xl bg-teal-500/20 border border-teal-500/40 text-teal-300 hover:bg-teal-500/30 transition-all cursor-pointer"
              title={isPlaying ? 'Pause auto-advance' : 'Play timeline auto-advance (Space)'}
            >
              {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            </button>

            <button
              onClick={() => setPlaySpeed(playSpeed === 1 ? 2 : playSpeed === 2 ? 4 : 1)}
              className="px-2 py-0.5 rounded-lg bg-black/40 border border-white/10 text-[10px] font-mono font-bold text-amber-400 cursor-pointer"
            >
              {playSpeed}x
            </button>
          </div>

          <div className="flex-1 flex items-center gap-3">
            <span className="text-[10px] font-mono text-slate-400">1</span>
            <input
              type="range"
              min="0"
              max={L - 1}
              value={scrubberIndex ?? 0}
              onChange={(e) => {
                const idx = parseInt(e.target.value, 10);
                setScrubberIndex(idx);
                handleHoverResidue(idx);
              }}
              className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-teal-400"
            />
            <span className="text-[10px] font-mono text-slate-400">{L}</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded-md font-mono text-[9px] font-bold text-amber-400 bg-amber-400/10 border border-amber-400/20">
              Schematic 3D Ribbon (Illustrative 2° structure)
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProteinStructure3D;
