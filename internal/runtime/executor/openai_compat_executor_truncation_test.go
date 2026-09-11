package executor

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
)

func truncatedStreamOpts(payload []byte) cliproxyexecutor.Options {
	return cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FromString("claude"),
		OriginalRequest: payload,
		Stream:          true,
	}
}

func truncatedStreamPayload() []byte {
	return []byte(`{"model":"claude-test","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
}

// Upstream closes with content but NO finish_reason and NO [DONE]: a
// truncation. The executor must surface a terminal error and must NOT emit
// message_stop (which downstream would read as a clean end_turn).
func TestOpenAICompatExecutor_TruncatedStreamSurfacesError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial answer\"},\"finish_reason\":null}]}\n\n"))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL + "/v1",
		"api_key":  "test",
	}}
	payload := truncatedStreamPayload()

	result, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "claude-test",
		Payload: payload,
	}, truncatedStreamOpts(payload))
	if err != nil {
		t.Fatalf("ExecuteStream() error = %v", err)
	}

	var combined strings.Builder
	var streamErr error
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			streamErr = chunk.Err
			continue
		}
		combined.Write(chunk.Payload)
	}
	if streamErr == nil {
		t.Fatalf("expected terminal stream error on truncation, got none; payloads=%q", combined.String())
	}
	if !strings.Contains(streamErr.Error(), "without terminal marker") {
		t.Fatalf("error = %q, want truncation marker message", streamErr.Error())
	}
	if strings.Contains(combined.String(), `"type":"message_stop"`) {
		t.Fatalf("truncated stream emitted message_stop (fake clean stop): %q", combined.String())
	}
}

// Upstream sends finish_reason + [DONE]: a real completion. No error, and
// message_stop must appear exactly once.
func TestOpenAICompatExecutor_CompletedStreamHasNoError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"full answer\"},\"finish_reason\":null}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"x\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL + "/v1",
		"api_key":  "test",
	}}
	payload := truncatedStreamPayload()

	result, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "claude-test",
		Payload: payload,
	}, truncatedStreamOpts(payload))
	if err != nil {
		t.Fatalf("ExecuteStream() error = %v", err)
	}

	var combined strings.Builder
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("unexpected chunk error on completed stream: %v", chunk.Err)
		}
		combined.Write(chunk.Payload)
	}
	if got := strings.Count(combined.String(), `"type":"message_stop"`); got != 1 {
		t.Fatalf("message_stop count = %d, want 1; payloads=%q", got, combined.String())
	}
	if !strings.Contains(combined.String(), `"stop_reason":"end_turn"`) {
		t.Fatalf("completed stream missing end_turn stop_reason; payloads=%q", combined.String())
	}
}
