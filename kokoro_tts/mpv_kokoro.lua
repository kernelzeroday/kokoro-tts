-- Loaded by kokoro-tts into mpv. The playlist lists every chunk of the
-- document up front, rendered or not. When mpv reaches (or the user skips
-- to) a chunk whose file does not exist yet, play looping silence instead and
-- reload the real chunk as soon as kokoro-tts finishes writing it. kokoro-tts
-- watches playlist-pos and renders from wherever the listener is.
local utils = require "mp.utils"

local dir = mp.get_opt("kokoro-dir") or ""
local silence = mp.get_opt("kokoro-silence")
local waiting = nil -- path of the chunk the placeholder stands in for

local function ours(path)
    return dir ~= "" and path:sub(1, #dir) == dir and path:match("%.flac$")
end

mp.add_hook("on_load", 50, function()
    waiting = nil
    local path = mp.get_property("stream-open-filename", "")
    if silence and ours(path) and not utils.file_info(path) then
        waiting = path
        local n = mp.get_property("playlist-pos-1", "?")
        local total = mp.get_property("playlist-count", "?")
        mp.set_property("stream-open-filename", silence)
        mp.set_property("file-local-options/loop-file", "inf")
        mp.set_property("file-local-options/force-media-title",
            ("Rendering part %s/%s…"):format(n, total))
    end
end)

mp.add_periodic_timer(0.5, function()
    if waiting and utils.file_info(waiting) then
        waiting = nil
        mp.commandv("playlist-play-index", "current")
    end
end)

-- Jump 10 chunks at a time (PgUp/PgDn are chapter seeks by default, and these
-- files have no chapters). A user input.conf binding still wins.
local function jump(delta)
    local pos = mp.get_property_number("playlist-pos", 0)
    local count = mp.get_property_number("playlist-count", 1)
    local target = math.max(0, math.min(count - 1, pos + delta))
    mp.commandv("playlist-play-index", tostring(target))
end
mp.add_key_binding("PGDWN", "kokoro-forward-10", function() jump(10) end)
mp.add_key_binding("PGUP", "kokoro-back-10", function() jump(-10) end)
