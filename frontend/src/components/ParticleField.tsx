import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';

/**
 * ParticleField — Cursor-reactive ambient particle background for secondary sections.
 *
 * Uses native Canvas2D (not Three.js) for lightweight rendering of small glowing dots
 * in Aurora accent colors that gently repel from the cursor position.
 *
 * - Performance-budgeted: 120 particles desktop, 40 mobile
 * - prefers-reduced-motion: static dots, no cursor reaction
 * - Disable on detected low-power devices
 */

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  baseX: number;
  baseY: number;
  radius: number;
  color: string;
  alpha: number;
}

const AURORA_COLORS = [
  'rgba(139, 92, 246, ',   // violet
  'rgba(0, 229, 204, ',    // teal
  'rgba(245, 158, 11, ',   // amber
  'rgba(168, 85, 247, ',   // purple
  'rgba(45, 212, 191, ',   // teal-light
];

export const ParticleField: React.FC<{ className?: string }> = ({ className }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const mouseRef = useRef({ x: -9999, y: -9999 });
  const particlesRef = useRef<Particle[]>([]);
  const [isVisible, setIsVisible] = useState(false);

  const reducedMotion = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }, []);

  const isMobile = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return window.innerWidth < 768 || navigator.maxTouchPoints > 1;
  }, []);

  const isLowPower = useMemo(() => {
    if (typeof navigator === 'undefined') return false;
    return (navigator.hardwareConcurrency || 4) <= 4;
  }, []);

  const particleCount = isMobile || isLowPower ? 40 : 120;
  const REPEL_RADIUS = 80;
  const REPEL_STRENGTH = 0.015;
  const RETURN_SPEED = 0.02;

  // Initialize particles
  const initParticles = useCallback((width: number, height: number) => {
    const particles: Particle[] = [];
    for (let i = 0; i < particleCount; i++) {
      const x = Math.random() * width;
      const y = Math.random() * height;
      particles.push({
        x, y,
        vx: 0, vy: 0,
        baseX: x, baseY: y,
        radius: 1.5 + Math.random() * 1.5,
        color: AURORA_COLORS[Math.floor(Math.random() * AURORA_COLORS.length)],
        alpha: 0.15 + Math.random() * 0.35,
      });
    }
    particlesRef.current = particles;
  }, [particleCount]);

  // IntersectionObserver for lazy activation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const observer = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold: 0.05 }
    );
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  // Mouse tracking
  useEffect(() => {
    if (reducedMotion || !isVisible) return;

    const handleMouse = (e: MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      mouseRef.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
    };

    const handleLeave = () => {
      mouseRef.current = { x: -9999, y: -9999 };
    };

    const canvas = canvasRef.current;
    if (canvas) {
      canvas.addEventListener('mousemove', handleMouse, { passive: true });
      canvas.addEventListener('mouseleave', handleLeave);
    }
    return () => {
      if (canvas) {
        canvas.removeEventListener('mousemove', handleMouse);
        canvas.removeEventListener('mouseleave', handleLeave);
      }
    };
  }, [reducedMotion, isVisible]);

  // Resize handler
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const w = parent.clientWidth;
      const h = parent.clientHeight;
      canvas.width = w;
      canvas.height = h;
      initParticles(w, h);
    };

    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, [initParticles]);

  // Animation loop
  useEffect(() => {
    if (!isVisible) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const animate = () => {
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;
      const particles = particlesRef.current;

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        if (!reducedMotion) {
          // Cursor repulsion
          const dx = p.x - mx;
          const dy = p.y - my;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < REPEL_RADIUS && dist > 0.1) {
            const force = (REPEL_RADIUS - dist) / REPEL_RADIUS * REPEL_STRENGTH;
            p.vx += (dx / dist) * force * REPEL_RADIUS;
            p.vy += (dy / dist) * force * REPEL_RADIUS;
          }

          // Spring back to base position
          p.vx += (p.baseX - p.x) * RETURN_SPEED;
          p.vy += (p.baseY - p.y) * RETURN_SPEED;

          // Damping
          p.vx *= 0.92;
          p.vy *= 0.92;

          p.x += p.vx;
          p.y += p.vy;
        }

        // Draw particle with glow
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = p.color + p.alpha + ')';
        ctx.fill();

        // Subtle glow effect
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = p.color + (p.alpha * 0.15) + ')';
        ctx.fill();
      }

      animFrameRef.current = requestAnimationFrame(animate);
    };

    animFrameRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [isVisible, reducedMotion]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'auto', // need pointer events for mouse tracking
        zIndex: 0,
      }}
      aria-hidden="true"
    />
  );
};

export default ParticleField;
