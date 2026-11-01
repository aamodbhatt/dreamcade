// Loads web/assets/dungeon_world.json (written by `dreamcade world`) into typed arrays.

export function parseWorld(json) {
  return {
    ...json,
    walls: Int16Array.from(json.walls),
    rooms: Int16Array.from(json.rooms),
    textures: Uint8Array.from(json.textures),
  };
}

export async function loadWorld(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status} (run \`dreamcade world\`)`);
  return parseWorld(await res.json());
}
