-- wrk -t2 -c100 -d30s -s strees_test_generate.lua http://0.0.0.0:8000/generate
counter = 0

-- Function to escape JSON string
function escape_json(s)
  -- Replace special characters with their escaped versions
  local escaped = s:gsub('\\', '\\\\')
                   :gsub('"', '\\"')
                   :gsub('\n', '\\n')
                   :gsub('\r', '\\r')
                   :gsub('\t', '\\t')
  return escaped
end

request = function()
  counter = counter + 1
  local prompt = string.format("User: Hello, this is user %d, are you there?\nAssistant: ", counter)
  local escaped_prompt = escape_json(prompt)
  local body = string.format('{"prompt": "%s", "max_tokens": 100, "temperature": 0.7, "output_scores": false}', escaped_prompt)
  wrk.headers["Content-Type"] = "application/json"
  wrk.headers["User-Agent"] = "stress_test/11.0.2"
  return wrk.format("POST", nil, nil, body)
end
