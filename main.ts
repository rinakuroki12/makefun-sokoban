import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const L = ['u', 'd', 'l', 'r'];

type Level = { w: number; n: number; floor: Uint8Array; goal: Uint8Array; goals: number[]; boxes: number[]; player: number };

function parse(text: string): Level {
  const rows = text.split('\n').filter(Boolean), w = Math.max(...rows.map(x => x.length)), n = w * rows.length;
  const floor = new Uint8Array(n), goal = new Uint8Array(n), boxes: number[] = [], goals: number[] = [];
  let player = 0;
  for (let y = 0; y < rows.length; y++) for (let x = 0; x < w; x++) {
    const c = rows[y][x] ?? '#', p = y * w + x;
    if (c !== '#') floor[p] = 1;
    if ('.*+'.includes(c)) { goal[p] = 1; goals.push(p); }
    if ('$*'.includes(c)) boxes.push(p);
    if ('@+'.includes(c)) player = p;
  }
  boxes.sort((a, b) => a - b);
  return { w, n, floor, goal, goals, boxes, player };
}

function search(level: Level, initial: number[], goals: number[], start: number, fixed: number[] = [], layered = false): string {
  const { w, n, floor } = level, count = initial.length, goal = new Uint8Array(n);
  for (const g of goals) goal[g] = 1;
  if (!count || count !== goals.length) return '';
  const D = [-w, w, -1, 1];

  // Reverse pushes give exact static box-to-goal lower bounds and identify non-goal dead squares.
  const gd = goals.map(() => new Int16Array(n).fill(30000));
  for (let gi = 0; gi < goals.length; gi++) {
    const q = [goals[gi]]; gd[gi][q[0]] = 0;
    for (let h = 0; h < q.length; h++) {
      const v = q[h], y = (v / w) | 0;
      for (const d of D) {
        const p = v + d, stand = p + d;
        if (p < 0 || p >= n || stand < 0 || stand >= n || !floor[p] || !floor[stand] || ((d === -1 || d === 1) && ((p / w) | 0) !== y)) continue;
        if (gd[gi][p] > gd[gi][v] + 1) { gd[gi][p] = gd[gi][v] + 1; q.push(p); }
      }
    }
  }
  const anyDist = new Int16Array(n).fill(30000);
  for (let p = 0; p < n; p++) for (let j = 0; j < count; j++) if (gd[j][p] < anyDist[p]) anyDist[p] = gd[j][p];

  const hcache = new Map<string, number>();
  function heuristic(bs: number[], bk: string): number {
    const cached = hcache.get(bk); if (cached !== undefined) return cached;
    // Hungarian minimum matching; the estimate is admissible and handles competing goals.
    const u = new Int32Array(count + 1), v = new Int32Array(count + 1), p = new Int32Array(count + 1), way = new Int32Array(count + 1);
    for (let i = 1; i <= count; i++) {
      p[0] = i; let j0 = 0;
      const minv = new Int32Array(count + 1).fill(1000000), used = new Uint8Array(count + 1);
      do {
        used[j0] = 1; const i0 = p[j0]!; let delta = 1000000, j1 = 0;
        for (let j = 1; j <= count; j++) if (!used[j]) {
          const c = gd[j - 1][bs[i0 - 1]]!;
          const cur = c - u[i0]! - v[j]!;
          if (cur < minv[j]!) { minv[j] = cur; way[j] = j0; }
          if (minv[j]! < delta) { delta = minv[j]!; j1 = j; }
        }
        for (let j = 0; j <= count; j++) if (used[j]) { u[p[j]!] += delta; v[j] -= delta; } else minv[j] -= delta;
        j0 = j1;
      } while (p[j0] !== 0);
      do { const j1 = way[j0]!; p[j0] = p[j1]!; j0 = j1; } while (j0);
    }
    let result = -v[0]!; hcache.set(bk, result); return result;
  }
  const boxKey = (bs: number[]) => String.fromCharCode(...bs);
  const keyOf = (region: number, bk: string) => String.fromCharCode(region) + bk;
  function regionOf(origin: number, bs: number[]): number {
    const blocked = new Uint8Array(n); for (const b of bs) blocked[b] = 1; for (const b of fixed) blocked[b] = 1;
    const seen = new Uint8Array(n), q = [origin]; seen[origin] = 1; let min = origin;
    for (let h = 0; h < q.length; h++) {
      const x = q[h]!; if (x < min) min = x;
      for (let k = 0; k < 4; k++) { const y = x + D[k]!; if (y < 0 || y >= n || !floor[y] || blocked[y] || seen[y] || (k >= 2 && ((x / w) | 0) !== ((y / w) | 0))) continue; seen[y] = 1; q.push(y); }
    }
    return min;
  }

  const players: number[] = [start], allBoxes: number[][] = [initial], costs: number[] = [0], parents: number[] = [-1], paths: string[] = [''];
  const startBk = boxKey(initial), startKey = keyOf(regionOf(start, initial), startBk), best = new Map<string, number>([[startKey, 0]]);
  const heap: number[] = [], fifo: number[] = layered ? [0] : []; let head = 0;
  const less = (a: number, b: number) => costs[a]! + heuristic(allBoxes[a]!, boxKey(allBoxes[a]!)) < costs[b]! + heuristic(allBoxes[b]!, boxKey(allBoxes[b]!));
  function push(id: number) { let i = heap.length; heap.push(id); while (i) { const a = (i - 1) >> 1; if (!less(heap[i]!, heap[a]!)) break; [heap[i], heap[a]] = [heap[a]!, heap[i]!]; i = a; } }
  function pop() { const top = heap[0]!, last = heap.pop()!; if (heap.length) { heap[0] = last; for (let i = 0;;) { let c = i * 2 + 1; if (c >= heap.length) break; if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++; if (!less(heap[c]!, heap[i]!)) break; [heap[i], heap[c]] = [heap[c]!, heap[i]!]; i = c; } } return top; }
  if (!layered) push(0);

  while (layered ? head < fifo.length : heap.length > 0) {
    const s = layered ? fifo[head++]! : pop(), bs = allBoxes[s]!, bk = boxKey(bs), b0 = best.get(keyOf(regionOf(players[s]!, bs), bk));
    if (b0 !== s) continue;
    if (bs.every(x => goal[x])) {
      let out = ''; for (let t = s; t > 0; t = parents[t]!) out = paths[t]! + out;
      return out;
    }
    const p0 = players[s]!, occupied = new Uint8Array(n); for (const b of bs) occupied[b] = 1; for (const b of fixed) occupied[b] = 1;
    const prev = new Int32Array(n).fill(-2), step = new Int8Array(n), q = [p0]; prev[p0] = -1;
    for (let h = 0; h < q.length; h++) {
      const x = q[h]!;
      for (let k = 0; k < 4; k++) {
        const d = D[k]!, y = x + d;
        if (y < 0 || y >= n || !floor[y] || (k >= 2 && ((y / w) | 0) !== ((x / w) | 0)) || occupied[y] || prev[y] !== -2) continue;
        prev[y] = x; step[y] = k; q.push(y);
      }
    }
    let region = p0; for (const x of q) if (x < region) region = x;
    const parentPath = (to: number) => { let s = ''; while (to !== p0) { const k = step[to]!; s = L[k]! + s; to = prev[to]!; } return s; };
    for (let bi = 0; bi < count; bi++) {
      const b = bs[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = D[k]!, stand = b - d, dest = b + d;
        if (stand < 0 || stand >= n || dest < 0 || dest >= n || prev[stand] === -2 || !floor[dest] || occupied[dest] || (k >= 2 && (((b / w) | 0) !== ((dest / w) | 0)))) continue;
        if (anyDist[dest] === 30000) continue;
        const next = bs.slice(); next[bi] = dest; next.sort((a, b) => a - b);
        let dead = false;
        // A fully blocked 2x2 containing an off-goal box is permanently frozen.
        for (const c of [dest, dest - 1, dest - w, dest - w - 1]) {
          const x = c % w, y = (c / w) | 0; if (x < 0 || x + 1 >= w || y < 0 || y + 1 >= n / w) continue;
          const sq = [c, c + 1, c + w, c + w + 1];
          if (sq.every(z => !floor[z] || next.includes(z)) && sq.some(z => next.includes(z) && !goal[z])) { dead = true; break; }
        }
        if (dead) continue;
        const nbk = boxKey(next), np = b, nk = keyOf(regionOf(np, next), nbk), old = best.get(nk), ng = costs[s]! + 1;
        if (old !== undefined && costs[old]! <= ng) continue;
        const id = players.length;
        players.push(np); allBoxes.push(next); costs.push(ng); parents.push(s); paths.push(parentPath(stand) + L[k]!.toUpperCase()); best.set(nk, id); if (layered) fifo.push(id); else push(id);
      }
    }
  }
  return '';
}

function rooms(level: Level): { boxes: number[]; goals: number[] }[] | null {
  const { w, n, floor, boxes, goals } = level, adj: number[][] = Array.from({ length: n }, () => []);
  for (let p = 0; p < n; p++) if (floor[p]) for (const d of [-w, w, -1, 1]) {
    const q = p + d; if (q >= 0 && q < n && floor[q] && (!(d === -1 || d === 1) || ((p / w) | 0) === ((q / w) | 0))) adj[p].push(q);
  }
  const disc = new Int32Array(n), low = new Int32Array(n), stack: [number, number][] = [], comps: number[][] = []; let time = 0;
  function dfs(v: number, par: number) {
    disc[v] = low[v] = ++time;
    for (const u of adj[v]!) {
      if (u === par) continue;
      if (!disc[u]) {
        stack.push([v, u]); dfs(u, v); low[v] = Math.min(low[v]!, low[u]!);
        if (low[u]! >= disc[v]!) { const c = new Set<number>(); let e: [number, number]; do { e = stack.pop()!; c.add(e[0]); c.add(e[1]); } while (e[0] !== v || e[1] !== u); comps.push([...c]); }
      } else if (disc[u]! < disc[v]!) { stack.push([v, u]); low[v] = Math.min(low[v]!, disc[u]!); }
    }
  }
  for (let p = 0; p < n; p++) if (floor[p] && !disc[p]) dfs(p, -1);
  const big = comps.filter(c => c.length >= 14); if (big.length < 2 || big.length > 4) return null;
  const dists = big.map(c => { const d = new Int16Array(n).fill(30000), q = c.slice(); for (const p of c) d[p] = 0; for (let h = 0; h < q.length; h++) for (const u of adj[q[h]!]!) if (d[u]! > d[q[h]!]! + 1) { d[u] = d[q[h]!]! + 1; q.push(u); } return d; });
  const groups = big.map(() => ({ boxes: [] as number[], goals: [] as number[] }));
  function assign(p: number, out: 'boxes' | 'goals') {
    let bi = -1, bd = 30000, tie = false;
    for (let i = 0; i < big.length; i++) { const x = dists[i]![p]!; if (x < bd) { bd = x; bi = i; tie = false; } else if (x === bd) tie = true; }
    if (bi < 0 || tie) return false; groups[bi]![out].push(p); return true;
  }
  for (const b of boxes) if (!assign(b, 'boxes')) return null;
  for (const g of goals) if (!assign(g, 'goals')) return null;
  if (groups.some(g => g.boxes.length !== g.goals.length || !g.boxes.length || g.boxes.length > 5)) return null;
  return groups;
}

function replay(level: Level, moves: string, boxes: number[], player: number): { boxes: number[]; player: number } {
  const { w, floor } = level, ds: Record<string, number> = { u: -w, d: w, l: -1, r: 1 }, bs = new Set(boxes);
  for (const ch of moves) { const d = ds[ch.toLowerCase()]!, q = player + d; if (!floor[q] || ((d === -1 || d === 1) && ((q / w) | 0) !== ((player / w) | 0))) return { boxes, player }; if (bs.has(q)) { const z = q + d; if (!floor[z] || bs.has(z)) return { boxes, player }; bs.delete(q); bs.add(z); } player = q; }
  return { boxes: [...bs].sort((a, b) => a - b), player };
}

function solve(text: string): string {
  const level = parse(text), split = rooms(level);
  if (split) {
    const order: number[][] = [];
    function perm(a: number[], left: number[]) { if (!left.length) { order.push(a); return; } for (let i = 0; i < left.length; i++) perm([...a, left[i]!], left.filter((_, j) => j !== i)); }
    perm([], split.map((_, i) => i));
    for (const seq of order) {
      let bs = level.boxes.slice(), p = level.player, out = '', ok = true;
      for (const ri of seq) {
        const group = split[ri]!, selected = new Set(group.boxes), moving = bs.filter(b => selected.has(b)), fixed = bs.filter(b => !selected.has(b));
        const path = search(level, moving, group.goals, p, fixed, true);
        if (!path) { ok = false; break; }
        const next = replay(level, path, bs, p); bs = next.boxes; p = next.player; out += path;
      }
      if (ok && bs.every(b => level.goal[b])) return out;
    }
  }
  return search(level, level.boxes, level.goals, level.player);
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
