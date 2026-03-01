import { useState, useEffect, useRef, useCallback } from "react";

// ─── Constants ───────────────────────────────────────────────────────────────
const CELL = {
  EMPTY: 0,
  WALL: 1,
  START: 2,
  GOAL: 3,
  FRONTIER: 4,
  VISITED: 5,
  PATH: 6,
  AGENT: 7,
};

const COLORS = {
  [CELL.EMPTY]: "#0f1117",
  [CELL.WALL]: "#374151",
  [CELL.START]: "#10b981",
  [CELL.GOAL]: "#f59e0b",
  [CELL.FRONTIER]: "#fde68a",
  [CELL.VISITED]: "#3b82f6",
  [CELL.PATH]: "#34d399",
  [CELL.AGENT]: "#f43f5e",
};

const HEURISTICS = {
  manhattan: (a, b) => Math.abs(a.r - b.r) + Math.abs(a.c - b.c),
  euclidean: (a, b) => Math.sqrt((a.r - b.r) ** 2 + (a.c - b.c) ** 2),
  chebyshev: (a, b) => Math.max(Math.abs(a.r - b.r), Math.abs(a.c - b.c)),
  octile: (a, b) => {
    const dx = Math.abs(a.r - b.r), dy = Math.abs(a.c - b.c);
    return Math.max(dx, dy) + (Math.sqrt(2) - 1) * Math.min(dx, dy);
  },
};

// ─── Priority Queue ───────────────────────────────────────────────────────────
class MinHeap {
  constructor() { this.data = []; }
  push(item) {
    this.data.push(item);
    this._bubbleUp(this.data.length - 1);
  }
  pop() {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) { this.data[0] = last; this._sinkDown(0); }
    return top;
  }
  _bubbleUp(i) {
    while (i > 0) {
      const p = Math.floor((i - 1) / 2);
      if (this.data[p].f <= this.data[i].f) break;
      [this.data[p], this.data[i]] = [this.data[i], this.data[p]];
      i = p;
    }
  }
  _sinkDown(i) {
    const n = this.data.length;
    while (true) {
      let min = i, l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.data[l].f < this.data[min].f) min = l;
      if (r < n && this.data[r].f < this.data[min].f) min = r;
      if (min === i) break;
      [this.data[min], this.data[i]] = [this.data[i], this.data[min]];
      i = min;
    }
  }
  get size() { return this.data.length; }
}

// ─── Pathfinding Algorithms ───────────────────────────────────────────────────
function getNeighbors(r, c, rows, cols, walls) {
  const dirs = [[-1,0],[1,0],[0,-1],[0,1]];
  return dirs
    .map(([dr,dc]) => [r+dr, c+dc])
    .filter(([nr,nc]) => nr>=0 && nr<rows && nc>=0 && nc<cols && !walls[nr][nc]);
}

function reconstructPath(cameFrom, goal) {
  const path = [];
  let cur = `${goal.r},${goal.c}`;
  while (cameFrom[cur]) {
    const [r,c] = cur.split(",").map(Number);
    path.unshift({r,c});
    cur = cameFrom[cur];
  }
  const [sr,sc] = cur.split(",").map(Number);
  path.unshift({r:sr,c:sc});
  return path;
}

function runSearch(algorithm, heuristic, start, goal, walls, rows, cols) {
  const t0 = performance.now();
  const h = HEURISTICS[heuristic];
  const pq = new MinHeap();
  const visited = new Set();
  const frontier = new Set();
  const cameFrom = {};
  const gScore = {};
  const key = (r,c) => `${r},${c}`;
  const startKey = key(start.r, start.c);
  gScore[startKey] = 0;
  const fStart = algorithm === "astar" ? h(start, goal) : h(start, goal);
  pq.push({ f: fStart, r: start.r, c: start.c });
  frontier.add(startKey);
  const visitedList = [];
  const frontierSnapshots = [];

  while (pq.size > 0) {
    const { r, c } = pq.pop();
    const k = key(r, c);
    frontier.delete(k);
    if (visited.has(k)) continue;
    visited.add(k);
    visitedList.push({r,c});

    if (r === goal.r && c === goal.c) {
      const path = reconstructPath(cameFrom, goal);
      return {
        path,
        visited: visitedList,
        nodesVisited: visited.size,
        pathCost: path.length - 1,
        time: performance.now() - t0,
        found: true,
      };
    }

    for (const [nr,nc] of getNeighbors(r, c, rows, cols, walls)) {
      const nk = key(nr,nc);
      if (visited.has(nk)) continue;
      const newG = (gScore[k] || 0) + 1;
      if (gScore[nk] !== undefined && gScore[nk] <= newG) continue;
      gScore[nk] = newG;
      cameFrom[nk] = k;
      const hVal = h({r:nr,c:nc}, goal);
      const f = algorithm === "astar" ? newG + hVal : hVal;
      pq.push({ f, r:nr, c:nc });
      frontier.add(nk);
    }
  }

  return { path: [], visited: visitedList, nodesVisited: visited.size, pathCost: 0, time: performance.now() - t0, found: false };
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function PathfindingAgent() {
  const [rows, setRows] = useState(20);
  const [cols, setCols] = useState(30);
  const [density, setDensity] = useState(30);
  const [algorithm, setAlgorithm] = useState("astar");
  const [heuristic, setHeuristic] = useState("manhattan");
  const [animSpeed, setAnimSpeed] = useState(20);
  const [dynamicMode, setDynamicMode] = useState(false);
  const [spawnProb, setSpawnProb] = useState(3);
  const [drawMode, setDrawMode] = useState("wall"); // wall | start | goal | erase

  const [grid, setGrid] = useState(null);
  const [walls, setWalls] = useState(null);
  const [start, setStart] = useState({ r: 2, c: 2 });
  const [goal, setGoal] = useState({ r: 17, c: 27 });
  const [status, setStatus] = useState("idle"); // idle | running | done | no_path
  const [metrics, setMetrics] = useState({ nodesVisited: 0, pathCost: 0, time: 0 });
  const [agentPos, setAgentPos] = useState(null);
  const [currentPath, setCurrentPath] = useState([]);

  const animRef = useRef(null);
  const dynamicRef = useRef(null);
  const wallsRef = useRef(null);
  const pathRef = useRef([]);
  const agentRef = useRef(null);
  const startRef = useRef(start);
  const goalRef = useRef(goal);
  const statusRef = useRef("idle");

  // keep refs in sync
  useEffect(() => { startRef.current = start; }, [start]);
  useEffect(() => { goalRef.current = goal; }, [goal]);
  useEffect(() => { statusRef.current = status; }, [status]);

  const initGrid = useCallback((r, c, s, g, existingWalls) => {
    const w = existingWalls || Array.from({length:r}, () => new Array(c).fill(false));
    wallsRef.current = w.map(row => [...row]);
    const g2 = Array.from({length:r}, (_,ri) =>
      Array.from({length:c}, (_,ci) => {
        if (ri===s.r && ci===s.c) return CELL.START;
        if (ri===g.r && ci===g.c) return CELL.GOAL;
        if (w[ri][ci]) return CELL.WALL;
        return CELL.EMPTY;
      })
    );
    setGrid(g2);
    setWalls(w.map(row=>[...row]));
    setAgentPos(null);
    setCurrentPath([]);
    pathRef.current = [];
    setStatus("idle");
    setMetrics({ nodesVisited:0, pathCost:0, time:0 });
  }, []);

  useEffect(() => {
    initGrid(rows, cols, start, goal, null);
  }, []);

  const generateMaze = () => {
    stopAll();
    const w = Array.from({length:rows}, (_,r) =>
      Array.from({length:cols}, (_,c) => {
        if ((r===start.r && c===start.c)||(r===goal.r && c===goal.c)) return false;
        return Math.random() * 100 < density;
      })
    );
    initGrid(rows, cols, start, goal, w);
  };

  const clearGrid = () => {
    stopAll();
    initGrid(rows, cols, start, goal, null);
  };

  const applyNewSize = () => {
    stopAll();
    const nr = Math.max(5, Math.min(40, rows));
    const nc = Math.max(5, Math.min(60, cols));
    const ns = { r: Math.min(start.r, nr-1), c: Math.min(start.c, nc-1) };
    const ng = { r: Math.min(goal.r, nr-1), c: Math.min(goal.c, nc-1) };
    setStart(ns); setGoal(ng);
    initGrid(nr, nc, ns, ng, null);
  };

  const stopAll = () => {
    if (animRef.current) clearTimeout(animRef.current);
    if (dynamicRef.current) clearInterval(dynamicRef.current);
    animRef.current = null;
    dynamicRef.current = null;
    statusRef.current = "idle";
    setStatus("idle");
  };

  // ─── Visualize Search ──────────────────────────────────────────────────────
  const visualizeSearch = (fromPos, currentWalls, onComplete) => {
    const s = fromPos || startRef.current;
    const g = goalRef.current;
    const r = currentWalls || wallsRef.current;

    const result = runSearch(algorithm, heuristic, s, g, r, rows, cols);

    let step = 0;
    const allSteps = [
      ...result.visited.map(n => ({ type:"visited", node:n })),
      ...result.path.map(n => ({ type:"path", node:n })),
    ];

    setMetrics({ nodesVisited: result.nodesVisited, pathCost: result.pathCost, time: result.time.toFixed(2) });

    const animate = () => {
      if (step >= allSteps.length) {
        if (!result.found) {
          setStatus("no_path");
          statusRef.current = "no_path";
        } else {
          pathRef.current = result.path;
          setCurrentPath(result.path);
          if (onComplete) onComplete(result.path);
        }
        return;
      }

      const { type, node } = allSteps[step];
      step++;

      setGrid(prev => {
        if (!prev) return prev;
        const ng = prev.map(row => [...row]);
        const { r: nr, c: nc } = node;
        if (ng[nr][nc] !== CELL.START && ng[nr][nc] !== CELL.GOAL) {
          ng[nr][nc] = type === "visited" ? CELL.VISITED : CELL.PATH;
        }
        return ng;
      });

      animRef.current = setTimeout(animate, 1000 / animSpeed);
    };

    animate();
  };

  // ─── Agent Movement ────────────────────────────────────────────────────────
  const moveAgent = (path, walls) => {
    if (!path || path.length < 2) {
      setStatus("done");
      statusRef.current = "done";
      if (dynamicRef.current) { clearInterval(dynamicRef.current); dynamicRef.current = null; }
      return;
    }

    let idx = 1;

    const step = () => {
      if (statusRef.current !== "running") return;

      // Check if next step is blocked
      const nextNode = path[idx];
      if (wallsRef.current[nextNode.r][nextNode.c]) {
        // Replan from current position
        const curPos = path[idx - 1];
        clearInterval(dynamicRef.current);
        dynamicRef.current = null;
        replan(curPos);
        return;
      }

      const curPos = path[idx];
      setAgentPos({ ...curPos });
      agentRef.current = curPos;

      setGrid(prev => {
        if (!prev) return prev;
        const ng = prev.map(row => [...row]);
        // Clear old agent
        if (idx > 1) {
          const prev2 = path[idx - 1];
          if (ng[prev2.r][prev2.c] === CELL.AGENT) ng[prev2.r][prev2.c] = CELL.PATH;
        }
        if (ng[curPos.r][curPos.c] !== CELL.GOAL) ng[curPos.r][curPos.c] = CELL.AGENT;
        return ng;
      });

      if (curPos.r === goalRef.current.r && curPos.c === goalRef.current.c) {
        setStatus("done");
        statusRef.current = "done";
        clearInterval(dynamicRef.current);
        dynamicRef.current = null;
        return;
      }

      idx++;
      if (idx >= path.length) {
        setStatus("done");
        statusRef.current = "done";
        clearInterval(dynamicRef.current);
        dynamicRef.current = null;
      }
    };

    dynamicRef.current = setInterval(step, 1000 / Math.max(1, animSpeed / 3));
  };

  const replan = (fromPos) => {
    setGrid(prev => {
      if (!prev) return prev;
      const ng = prev.map(row => [...row]);
      // Reset visualized cells
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++)
          if (ng[r][c] === CELL.VISITED || ng[r][c] === CELL.PATH || ng[r][c] === CELL.FRONTIER) {
            ng[r][c] = wallsRef.current[r][c] ? CELL.WALL : CELL.EMPTY;
          }
      ng[startRef.current.r][startRef.current.c] = CELL.START;
      ng[goalRef.current.r][goalRef.current.c] = CELL.GOAL;
      return ng;
    });

    visualizeSearch(fromPos, wallsRef.current, (newPath) => {
      if (newPath && newPath.length > 1) {
        moveAgent(newPath, wallsRef.current);
      } else {
        setStatus("no_path");
        statusRef.current = "no_path";
      }
    });
  };

  // ─── Dynamic Obstacle Spawner ──────────────────────────────────────────────
  const startDynamicSpawner = () => {
    const spawner = setInterval(() => {
      if (statusRef.current !== "running") return;
      if (Math.random() * 100 > spawnProb) return;

      // Pick random empty cell (not start, goal, or current path)
      const r2 = Math.floor(Math.random() * rows);
      const c2 = Math.floor(Math.random() * cols);
      if (wallsRef.current[r2][c2]) return;
      if (r2 === startRef.current.r && c2 === startRef.current.c) return;
      if (r2 === goalRef.current.r && c2 === goalRef.current.c) return;
      if (agentRef.current && r2 === agentRef.current.r && c2 === agentRef.current.c) return;

      // Place wall
      wallsRef.current[r2][c2] = true;
      setWalls(w => { const nw = w.map(row=>[...row]); nw[r2][c2] = true; return nw; });
      setGrid(prev => {
        if (!prev) return prev;
        const ng = prev.map(row=>[...row]);
        if (ng[r2][c2] === CELL.EMPTY || ng[r2][c2] === CELL.VISITED || ng[r2][c2] === CELL.FRONTIER) {
          ng[r2][c2] = CELL.WALL;
        }
        return ng;
      });

      // Check if on current path
      const onPath = pathRef.current.some(n => n.r === r2 && n.c === c2);
      if (onPath && agentRef.current) {
        clearInterval(dynamicRef.current);
        dynamicRef.current = null;
        clearInterval(spawner);
        replan(agentRef.current);
      }
    }, 300);

    return spawner;
  };

  // ─── Start Search ──────────────────────────────────────────────────────────
  const startSearch = () => {
    stopAll();
    setStatus("running");
    statusRef.current = "running";
    agentRef.current = start;
    setAgentPos(null);

    // Reset grid visuals
    setGrid(prev => {
      if (!prev) return prev;
      return prev.map((row, r) => row.map((cell, c) => {
        if (cell === CELL.VISITED || cell === CELL.PATH || cell === CELL.FRONTIER || cell === CELL.AGENT)
          return wallsRef.current[r][c] ? CELL.WALL : CELL.EMPTY;
        return cell;
      }));
    });

    visualizeSearch(start, wallsRef.current, (path) => {
      if (dynamicMode) {
        const spawnerInterval = startDynamicSpawner();
        moveAgent(path, wallsRef.current);
        // store spawner to clear later - hacky but functional
        setTimeout(() => {
          if (statusRef.current === "done" || statusRef.current === "no_path") {
            clearInterval(spawnerInterval);
          }
        }, 60000);
      } else {
        moveAgent(path, wallsRef.current);
      }
    });
  };

  // ─── Cell Interaction ──────────────────────────────────────────────────────
  const isDrawing = useRef(false);

  const handleCellInteract = (r, c) => {
    if (status === "running") return;
    if (drawMode === "start") {
      const ns = { r, c };
      setStart(ns);
      startRef.current = ns;
      initGrid(rows, cols, ns, goal, wallsRef.current);
    } else if (drawMode === "goal") {
      const ng2 = { r, c };
      setGoal(ng2);
      goalRef.current = ng2;
      initGrid(rows, cols, start, ng2, wallsRef.current);
    } else if (drawMode === "wall") {
      if ((r===start.r&&c===start.c)||(r===goal.r&&c===goal.c)) return;
      const nw = wallsRef.current.map(row=>[...row]);
      nw[r][c] = true;
      wallsRef.current = nw;
      setWalls(nw);
      setGrid(prev => {
        if (!prev) return prev;
        const ng = prev.map(row=>[...row]);
        ng[r][c] = CELL.WALL;
        return ng;
      });
    } else if (drawMode === "erase") {
      if ((r===start.r&&c===start.c)||(r===goal.r&&c===goal.c)) return;
      const nw = wallsRef.current.map(row=>[...row]);
      nw[r][c] = false;
      wallsRef.current = nw;
      setWalls(nw);
      setGrid(prev => {
        if (!prev) return prev;
        const ng = prev.map(row=>[...row]);
        ng[r][c] = CELL.EMPTY;
        return ng;
      });
    }
  };

  const cellSize = Math.min(Math.floor(560 / cols), Math.floor(420 / rows), 28);

  const statusColor = {
    idle: "#6b7280",
    running: "#f59e0b",
    done: "#10b981",
    no_path: "#ef4444",
  }[status] || "#6b7280";

  const statusText = {
    idle: "READY",
    running: "SEARCHING...",
    done: "PATH FOUND",
    no_path: "NO PATH",
  }[status] || "READY";

  return (
    <div style={{
      minHeight: "100vh",
      background: "#070a0f",
      fontFamily: "'Courier New', monospace",
      color: "#e2e8f0",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "20px",
    }}>
      {/* Header */}
      <div style={{ textAlign:"center", marginBottom:"16px" }}>
        <div style={{ fontSize:"11px", letterSpacing:"0.4em", color:"#4ade80", marginBottom:"4px", textTransform:"uppercase" }}>
          Informed Search Algorithms
        </div>
        <h1 style={{ margin:0, fontSize:"26px", fontWeight:"900", letterSpacing:"0.1em", color:"#f1f5f9" }}>
          DYNAMIC PATHFINDING AGENT
        </h1>
        <div style={{ display:"flex", gap:"8px", justifyContent:"center", alignItems:"center", marginTop:"8px" }}>
          <span style={{ background: statusColor, color:"#000", padding:"2px 12px", fontSize:"10px", fontWeight:"700", letterSpacing:"0.2em", borderRadius:"2px" }}>
            {statusText}
          </span>
        </div>
      </div>

      <div style={{ display:"flex", gap:"16px", maxWidth:"1200px", width:"100%", flexWrap:"wrap", justifyContent:"center" }}>
        {/* ── Left Panel ── */}
        <div style={{ display:"flex", flexDirection:"column", gap:"12px", width:"220px", flexShrink:0 }}>
          {/* Grid Config */}
          <Panel title="GRID CONFIG">
            <Row label="Rows">
              <NumInput value={rows} onChange={setRows} min={5} max={40} />
            </Row>
            <Row label="Cols">
              <NumInput value={cols} onChange={setCols} min={5} max={60} />
            </Row>
            <BtnRow>
              <Btn onClick={applyNewSize} color="#6366f1">Apply Size</Btn>
            </BtnRow>
            <Row label="Density %">
              <NumInput value={density} onChange={setDensity} min={0} max={80} />
            </Row>
            <BtnRow>
              <Btn onClick={generateMaze} color="#8b5cf6">Gen Maze</Btn>
              <Btn onClick={clearGrid} color="#374151">Clear</Btn>
            </BtnRow>
          </Panel>

          {/* Algorithm */}
          <Panel title="ALGORITHM">
            <SelectRow label="Algo" value={algorithm} onChange={setAlgorithm} options={[
              {value:"astar", label:"A* Search"},
              {value:"gbfs", label:"Greedy BFS"},
            ]} />
            <SelectRow label="Heuristic" value={heuristic} onChange={setHeuristic} options={[
              {value:"manhattan", label:"Manhattan"},
              {value:"euclidean", label:"Euclidean"},
              {value:"chebyshev", label:"Chebyshev"},
              {value:"octile", label:"Octile"},
            ]} />
            <Row label="Speed">
              <input type="range" min={1} max={200} value={animSpeed}
                onChange={e => setAnimSpeed(Number(e.target.value))}
                style={{ width:"100%", accentColor:"#4ade80" }} />
            </Row>
          </Panel>

          {/* Drawing */}
          <Panel title="DRAW MODE">
            {["wall","erase","start","goal"].map(m => (
              <button key={m} onClick={() => setDrawMode(m)} style={{
                display:"block", width:"100%", marginBottom:"4px",
                background: drawMode===m ? "#4ade80" : "#1e293b",
                color: drawMode===m ? "#000" : "#94a3b8",
                border: `1px solid ${drawMode===m ? "#4ade80" : "#334155"}`,
                padding:"5px", cursor:"pointer", fontSize:"11px",
                fontFamily:"'Courier New', monospace", letterSpacing:"0.1em",
                fontWeight: drawMode===m ? "700" : "400",
                borderRadius:"3px",
              }}>
                {m === "wall" ? "🧱 PLACE WALL" : m === "erase" ? "🗑 ERASE" : m === "start" ? "🟢 SET START" : "🟡 SET GOAL"}
              </button>
            ))}
          </Panel>

          {/* Dynamic */}
          <Panel title="DYNAMIC MODE">
            <label style={{ display:"flex", alignItems:"center", gap:"8px", cursor:"pointer", fontSize:"12px" }}>
              <input type="checkbox" checked={dynamicMode} onChange={e => setDynamicMode(e.target.checked)}
                style={{ accentColor:"#f59e0b", width:"14px", height:"14px" }} />
              Enable Dynamic Obstacles
            </label>
            {dynamicMode && (
              <Row label="Spawn %">
                <NumInput value={spawnProb} onChange={setSpawnProb} min={1} max={20} />
              </Row>
            )}
          </Panel>
        </div>

        {/* ── Grid ── */}
        <div style={{ display:"flex", flexDirection:"column", gap:"12px" }}>
          {/* Grid Canvas */}
          <div style={{
            background:"#0d1117",
            border:"1px solid #1e293b",
            padding:"10px",
            borderRadius:"6px",
            overflow:"auto",
          }}>
            <div
              style={{ display:"inline-grid", gridTemplateColumns:`repeat(${cols}, ${cellSize}px)`, gap:"1px", cursor:"crosshair" }}
              onMouseDown={() => { isDrawing.current = true; }}
              onMouseUp={() => { isDrawing.current = false; }}
              onMouseLeave={() => { isDrawing.current = false; }}
            >
              {grid && grid.map((row, r) =>
                row.map((cell, c) => (
                  <div
                    key={`${r}-${c}`}
                    onMouseDown={() => handleCellInteract(r, c)}
                    onMouseEnter={() => { if (isDrawing.current) handleCellInteract(r, c); }}
                    style={{
                      width: cellSize, height: cellSize,
                      background: agentPos && agentPos.r===r && agentPos.c===c ? COLORS[CELL.AGENT] : COLORS[cell] || COLORS[CELL.EMPTY],
                      borderRadius: cell === CELL.START || cell === CELL.GOAL ? "50%" : "1px",
                      transition: "background 0.1s",
                      boxSizing: "border-box",
                    }}
                  />
                ))
              )}
            </div>
          </div>

          {/* Controls */}
          <div style={{ display:"flex", gap:"10px", justifyContent:"center" }}>
            <button onClick={startSearch} disabled={status==="running"} style={{
              background: status==="running" ? "#374151" : "#4ade80",
              color: status==="running" ? "#6b7280" : "#000",
              border:"none", padding:"10px 28px", cursor: status==="running" ? "not-allowed" : "pointer",
              fontFamily:"'Courier New', monospace", fontSize:"13px", fontWeight:"700",
              letterSpacing:"0.15em", borderRadius:"4px",
            }}>
              {status==="running" ? "⏳ RUNNING..." : "▶ START SEARCH"}
            </button>
            <button onClick={stopAll} style={{
              background:"#ef4444", color:"#fff", border:"none", padding:"10px 20px",
              cursor:"pointer", fontFamily:"'Courier New', monospace", fontSize:"13px",
              fontWeight:"700", letterSpacing:"0.15em", borderRadius:"4px",
            }}>
              ■ STOP
            </button>
          </div>
        </div>

        {/* ── Right Panel ── */}
        <div style={{ display:"flex", flexDirection:"column", gap:"12px", width:"200px", flexShrink:0 }}>
          {/* Metrics */}
          <Panel title="METRICS">
            <MetricRow label="Nodes Visited" value={metrics.nodesVisited} color="#3b82f6" />
            <MetricRow label="Path Cost" value={metrics.pathCost} color="#34d399" />
            <MetricRow label="Exec Time (ms)" value={metrics.time} color="#f59e0b" />
            <MetricRow label="Algorithm" value={algorithm === "astar" ? "A*" : "GBFS"} color="#a78bfa" />
            <MetricRow label="Heuristic" value={heuristic.slice(0,8)} color="#fb7185" />
          </Panel>

          {/* Legend */}
          <Panel title="LEGEND">
            {[
              [COLORS[CELL.START], "Start Node"],
              [COLORS[CELL.GOAL], "Goal Node"],
              [COLORS[CELL.AGENT], "Agent"],
              [COLORS[CELL.FRONTIER], "Frontier"],
              [COLORS[CELL.VISITED], "Visited"],
              [COLORS[CELL.PATH], "Final Path"],
              [COLORS[CELL.WALL], "Obstacle"],
            ].map(([color, label]) => (
              <div key={label} style={{ display:"flex", alignItems:"center", gap:"8px", marginBottom:"5px" }}>
                <div style={{ width:"12px", height:"12px", background:color, borderRadius:"2px", flexShrink:0 }} />
                <span style={{ fontSize:"11px", color:"#94a3b8" }}>{label}</span>
              </div>
            ))}
          </Panel>

          {/* Info */}
          <Panel title="HOW TO USE">
            <p style={{ fontSize:"10px", color:"#64748b", lineHeight:"1.6", margin:0 }}>
              1. Set grid size & generate maze<br/>
              2. Choose algorithm & heuristic<br/>
              3. Click cells to edit obstacles<br/>
              4. Enable Dynamic Mode for live obstacles<br/>
              5. Press Start Search
            </p>
          </Panel>
        </div>
      </div>
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function Panel({ title, children }) {
  return (
    <div style={{ background:"#0d1117", border:"1px solid #1e293b", borderRadius:"6px", padding:"12px" }}>
      <div style={{ fontSize:"9px", letterSpacing:"0.3em", color:"#475569", marginBottom:"10px", fontWeight:"700" }}>{title}</div>
      {children}
    </div>
  );
}

function Row({ label, children }) {
  return (
    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:"6px" }}>
      <span style={{ fontSize:"11px", color:"#64748b" }}>{label}</span>
      {children}
    </div>
  );
}

function BtnRow({ children }) {
  return <div style={{ display:"flex", gap:"6px", marginBottom:"6px" }}>{children}</div>;
}

function Btn({ onClick, color, children }) {
  return (
    <button onClick={onClick} style={{
      flex:1, background:color, color:"#fff", border:"none", padding:"5px 0",
      cursor:"pointer", fontFamily:"'Courier New', monospace", fontSize:"10px",
      fontWeight:"700", borderRadius:"3px",
    }}>{children}</button>
  );
}

function NumInput({ value, onChange, min, max }) {
  return (
    <input type="number" value={value} min={min} max={max}
      onChange={e => onChange(Number(e.target.value))}
      style={{
        width:"60px", background:"#1e293b", color:"#e2e8f0", border:"1px solid #334155",
        padding:"3px 6px", fontFamily:"'Courier New', monospace", fontSize:"12px", borderRadius:"3px",
      }}
    />
  );
}

function SelectRow({ label, value, onChange, options }) {
  return (
    <Row label={label}>
      <select value={value} onChange={e => onChange(e.target.value)} style={{
        background:"#1e293b", color:"#e2e8f0", border:"1px solid #334155",
        padding:"3px 4px", fontFamily:"'Courier New', monospace", fontSize:"10px", borderRadius:"3px",
        width:"110px",
      }}>
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </Row>
  );
}

function MetricRow({ label, value, color }) {
  return (
    <div style={{ marginBottom:"8px" }}>
      <div style={{ fontSize:"9px", color:"#475569", letterSpacing:"0.1em", marginBottom:"2px" }}>{label.toUpperCase()}</div>
      <div style={{ fontSize:"18px", fontWeight:"900", color, lineHeight:1 }}>{value || 0}</div>
    </div>
  );
}
