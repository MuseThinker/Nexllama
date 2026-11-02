// Resident TTS worker for Nexllama.
//
// This is llama.cpp's tools/tts/tts.cpp with the model load kept alive and the
// one-shot prompt turned into a line protocol on stdin/stdout. The daemon owns
// the process: it starts it when a TTS model is loaded, keeps it resident while
// the model is loaded, and kills it on unload so the VRAM comes back.
//
// Protocol, binary frames on stdin/stdout:
//   daemon -> worker  "TTS1 <input> <lang> <voice> <out> <max_frames>\n" + bytes
//   worker -> daemon  "OK <bytes>\n"
//   worker -> daemon  "ERR <message>\n"
// The first line after a successful load is exactly READY; anything else means
// the worker could not start.
//
// Build against the same llama.cpp revision as the backend DLLs. See README.md
// in this directory for the compiler command.

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <fcntl.h>
#include <io.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

static void print_usage(int, char ** argv) {
    fprintf(stderr, "usage: %s -m backbone.gguf -mm mmproj.gguf [common options]\n", argv[0]);
    fprintf(stderr, "       requests are read as binary frames from stdin\n");
}

static void respond_ok(size_t bytes) {
    fprintf(stdout, "OK %zu\n", bytes);
    fflush(stdout);
}

static void respond_error(const std::string & message) {
    std::string one_line = message;
    for (char & c : one_line) {
        if (c == '\n' || c == '\r') {
            c = ' ';
        }
    }
    fprintf(stdout, "ERR %s\n", one_line.c_str());
    fflush(stdout);
}

int main(int argc, char ** argv) {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    common_params params;
    common_init();
    // The protocol owns stdout, and the daemon owns stderr. Disable llama.cpp's
    // stdio logger entirely; our own diagnostics use fprintf(stderr) below.
    common_log_set_stdo(common_log_main(), false);

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 1;
    }
    if (params.mmproj.path.empty()) {
        fprintf(stderr, "no mmproj provided, use --mmproj\n");
        return 1;
    }

    mtmd_helper_log_set(common_log_default_callback, nullptr);

    // Hidden states are what the audio helper consumes.
    params.embedding = true;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * lctx  = llama_init->context();
    common_sampler * smpl = llama_init->sampler(0);
    if (!model || !lctx || !smpl) {
        fprintf(stderr, "failed to init model/context\n");
        return 1;
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = params.mmproj_use_gpu;
    mtmd_params.device  = params.mmproj_device;
    mtmd::context_ptr mctx(mtmd_init_from_file(params.mmproj.path.c_str(), model, mtmd_params));
    if (!mctx) {
        fprintf(stderr, "failed to load mmproj %s\n", params.mmproj.path.c_str());
        return 1;
    }
    if (mtmd_gen_audio_get_info(mctx.get()).type == MTMD_GEN_AUDIO_TYPE_NONE) {
        fprintf(stderr, "mmproj does not support audio generation\n");
        return 1;
    }

    // The model is loaded; the daemon may now send requests.
    fputs("READY\n", stdout);
    fflush(stdout);

    std::string header;
    while (std::getline(std::cin, header)) {
        if (header.empty()) {
            continue;
        }

        std::istringstream hs(header);
        std::string magic;
        size_t input_len = 0;
        size_t language_len = 0;
        size_t voice_len = 0;
        size_t output_len = 0;
        int max_frames = params.n_predict > 0 ? params.n_predict : 512;
        if (!(hs >> magic >> input_len >> language_len >> voice_len >> output_len >> max_frames) ||
            magic != "TTS1") {
            respond_error("bad request");
            continue;
        }

        auto read_bytes = [&](size_t n, std::string & out) -> bool {
            out.resize(n);
            if (n == 0) {
                return true;
            }
            std::cin.read(&out[0], (std::streamsize) n);
            return std::cin.gcount() == (std::streamsize) n;
        };

        std::string input;
        std::string language;
        std::string voice;
        std::string output;
        if (!read_bytes(input_len, input) || !read_bytes(language_len, language) ||
            !read_bytes(voice_len, voice) || !read_bytes(output_len, output)) {
            respond_error("truncated request");
            break;
        }
        if (language.empty()) {
            language = "en";
        }
        if (input.empty()) {
            respond_error("input must not be empty");
            continue;
        }
        if (output.empty()) {
            respond_error("output path is required");
            continue;
        }

        // Every request starts from an empty sequence and a fresh sampler.
        llama_synchronize(lctx);
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_perf_context_reset(lctx);
        common_sampler_reset(smpl);

        mtmd::bitmap_ptr speaker_bitmap;
        if (!voice.empty()) {
            auto wrapper = mtmd_helper_bitmap_init_from_file(
                mctx.get(), voice.c_str(), false, mtmd_helper_init_opt_default());
            if (!wrapper.bitmap) {
                respond_error("failed to load speaker file " + voice);
                continue;
            }
            speaker_bitmap.reset(wrapper.bitmap);
        }

        mtmd_helper::gen_audio gen;
        gen.init(lctx, mctx.get());

        mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
        inp.seq_id      = 0;
        inp.prompt      = input.c_str();
        inp.prompt_len  = input.size();
        inp.speaker_ref = speaker_bitmap.get();
        inp.lang        = language.c_str();
        inp.top_k       = params.sampling.top_k;
        inp.top_p       = params.sampling.top_p;
        inp.seed        = params.sampling.seed;

        if (gen.set_input(&inp) != 0) {
            respond_error("set_input failed");
            continue;
        }

        bool failed = false;
        for (;;) {
            int32_t ret = gen.step_prompt(params.n_batch);
            if (ret < 0) {
                respond_error("prompt processing failed");
                failed = true;
                break;
            }
            if (ret == 0) {
                break;
            }
        }
        if (failed) {
            continue;
        }

        auto sample_semantic_code = [&]() -> llama_token {
            llama_token t = common_sampler_sample(smpl, lctx, -1);
            common_sampler_accept(smpl, t, true);
            return t;
        };

        const int max_new = max_frames > 0 ? max_frames : (params.n_predict > 0 ? params.n_predict : 512);
        int n_frames = 0;
        llama_token sampled = sample_semantic_code();
        const float * h_state = llama_get_embeddings_ith(lctx, -1);

        bool stop = false;
        while (!stop && n_frames < max_new) {
            const float * h_next = nullptr;
            if (gen.step_gen(sampled, h_state, &h_next, &stop) != 0) {
                respond_error("step_gen failed at frame " + std::to_string(n_frames));
                failed = true;
                break;
            }
            if (!h_next) {
                break;
            }
            n_frames++;
            h_state = h_next;
            sampled = sample_semantic_code();
        }
        if (failed) {
            continue;
        }

        int32_t sample_rate = 0;
        const char * data = nullptr;
        size_t data_len = 0;
        int64_t n_samples = 0;
        if (gen.get_output(&sample_rate, &data, &data_len, &n_samples) != 0) {
            respond_error("get_output failed");
            continue;
        }

        FILE * f = fopen(output.c_str(), "wb");
        if (!f) {
            respond_error("failed to open " + output);
            continue;
        }
        size_t written = fwrite(data, 1, data_len, f);
        fclose(f);
        if (written != data_len) {
            respond_error("failed to write " + output);
            continue;
        }

        respond_ok(written);
    }

    // stdin closed: the daemon is gone or unloaded us. Free VRAM and exit.
    llama_backend_free();
    return 0;
}
