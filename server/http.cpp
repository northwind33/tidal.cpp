#include "http.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

// A write to a closed connection must fail instead of raising SIGPIPE.
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace tidal::server {
namespace {

const char * reason(int status) {
    switch (status) {
        case 200: return "OK"; case 204: return "No Content"; case 400: return "Bad Request";
        case 401: return "Unauthorized"; case 404: return "Not Found"; case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large"; case 429: return "Too Many Requests";
        case 500: return "Internal Server Error"; case 501: return "Not Implemented";
        case 503: return "Service Unavailable"; default: return "Unknown";
    }
}
std::string lower(std::string text) {
    for (auto & c : text) if (c >= 'A' && c <= 'Z') c = char(c + 32);
    return text;
}
std::string_view strip(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) text.remove_suffix(1);
    return text;
}
std::string percent_decode(std::string_view text) {
    std::string out;
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned value = 0;
        if (text[i] == '%' && i + 2 < text.size() + 0 &&
            std::from_chars(text.data() + i + 1, text.data() + i + 3, value, 16).ptr == text.data() + i + 3) {
            out += char(value); i += 2;
        } else out += text[i] == '+' ? ' ' : text[i];
    }
    return out;
}

// Buffered socket reader.
struct Reader {
    int fd;
    std::string buffer;
    // Appends more bytes; false on end of stream, error or timeout.
    bool fill() {
        char chunk[16384];
        for (;;) {
            const ssize_t n = ::recv(fd, chunk, sizeof chunk, 0);
            if (n > 0) { buffer.append(chunk, size_t(n)); return true; }
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
    }
    bool line(std::string & out, size_t limit) {
        for (;;) {
            const auto end = buffer.find('\n');
            if (end != std::string::npos) {
                out.assign(buffer, 0, end);
                if (!out.empty() && out.back() == '\r') out.pop_back();
                buffer.erase(0, end + 1);
                return true;
            }
            if (buffer.size() > limit || !fill()) return false;
        }
    }
    bool exact(std::string & out, size_t size) {
        while (buffer.size() < size) if (!fill()) return false;
        out.append(buffer, 0, size);
        buffer.erase(0, size);
        return true;
    }
};

} // namespace

const std::string & Request::header(const std::string & name) const {
    static const std::string empty;
    const auto it = headers.find(name);
    return it == headers.end() ? empty : it->second;
}
std::string Request::parameter(std::string_view name) const {
    std::string_view rest = query;
    while (!rest.empty()) {
        const auto pair = rest.substr(0, rest.find('&'));
        rest.remove_prefix(std::min(rest.size(), pair.size() + 1));
        const auto equals = pair.find('=');
        if (pair.substr(0, equals) == name) return equals == std::string_view::npos ? std::string() : percent_decode(pair.substr(equals + 1));
    }
    return {};
}

bool Response::raw(std::string_view data) {
    while (!data.empty() && !failed_) {
        const ssize_t n = ::send(fd_, data.data(), data.size(), MSG_NOSIGNAL);
        if (n > 0) data.remove_prefix(size_t(n));
        else if (n < 0 && errno == EINTR) continue;
        else failed_ = true;
    }
    return !failed_;
}
void Response::header(std::string_view name, std::string_view value) {
    extra_.append(name); extra_ += ": "; extra_.append(value); extra_ += "\r\n";
}
std::string Response::head(int status, std::string_view content_type) const {
    std::string out = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\n";
    if (!content_type.empty()) { out += "Content-Type: "; out.append(content_type); out += "\r\n"; }
    out += "Access-Control-Allow-Origin: *\r\n";
    out += extra_;
    return out;
}
void Response::send(int status, std::string_view content_type, std::string_view body) {
    if (started_) return;
    started_ = true; status_ = status;
    std::string out = head(status, content_type);
    out += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    out += keep_alive_ ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n";
    if (!head_) out.append(body);
    raw(out);
}
bool Response::begin(int status, std::string_view content_type) {
    if (started_) return false;
    started_ = streaming_ = true; status_ = status;
    std::string out = head(status, content_type);
    out += "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n";
    if (chunked_) out += keep_alive_ ? "Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n" : "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    else { keep_alive_ = false; out += "Connection: close\r\n\r\n"; }
    return raw(out);
}
bool Response::write(std::string_view data) {
    if (!streaming_ || data.empty() || head_) return !failed_;
    if (!chunked_) return raw(data);
    char size[24];
    const int n = std::snprintf(size, sizeof size, "%zx\r\n", data.size());
    std::string out(size, size_t(n));
    out.append(data); out += "\r\n";
    return raw(out);
}
void Response::end() {
    if (!streaming_) return;
    if (chunked_ && !head_) raw("0\r\n\r\n");
    streaming_ = false;
}
bool Response::alive() {
    if (failed_) return false;
    char byte;
    const ssize_t n = ::recv(fd_, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) failed_ = true;
    return !failed_;
}

void HttpServer::route(const std::string & method, const std::string & path, Handler handler) {
    routes_.push_back({method, path, std::move(handler)});
}

void HttpServer::listen(const std::string & host, int port) {
    addrinfo hints{}, * found = nullptr;
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM; hints.ai_flags = AI_PASSIVE;
    const int status = getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &found);
    if (status) throw std::runtime_error("Cannot resolve " + host + ": " + gai_strerror(status));
    std::string error = "no address";
    for (auto * a = found; a && listen_fd_ < 0; a = a->ai_next) {
        const int fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) { error = std::strerror(errno); continue; }
        const int on = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
        if (::bind(fd, a->ai_addr, a->ai_addrlen) == 0 && ::listen(fd, 128) == 0) {
            sockaddr_storage bound{}; socklen_t length = sizeof bound;
            getsockname(fd, reinterpret_cast<sockaddr *>(&bound), &length);
            port_ = ntohs(bound.ss_family == AF_INET6 ? reinterpret_cast<sockaddr_in6 *>(&bound)->sin6_port
                                                      : reinterpret_cast<sockaddr_in *>(&bound)->sin_port);
            listen_fd_ = fd;
        } else { error = std::strerror(errno); ::close(fd); }
    }
    freeaddrinfo(found);
    if (listen_fd_ < 0) throw std::runtime_error("Cannot listen on " + host + ":" + std::to_string(port) + ": " + error);
}

void HttpServer::stop() {
    stopping_ = true;
    if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);
}

void HttpServer::run() {
    while (!stopping_) {
        // Polling bounds the wait, so stop() takes effect even where shutting
        // down a listening socket does not wake accept().
        pollfd waiting{listen_fd_, POLLIN, 0};
        if (::poll(&waiting, 1, 500) <= 0) continue;
        const int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) {
            if (stopping_) break;
            if (errno == EINTR || errno == ECONNABORTED) continue;
            if (errno == EMFILE || errno == ENFILE) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
            break;
        }
        const int on = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on);
#ifdef SO_NOSIGPIPE
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on);
#endif
        // An idle or stalled peer must not hold a thread (or the engine) forever.
        timeval receive{120, 0}, transmit{60, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &receive, sizeof receive);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &transmit, sizeof transmit);
        if (connections_ >= 256) {
            Response busy(fd, false, false);
            busy.send(503, "text/plain", "Too many connections\n");
            ::close(fd);
            continue;
        }
        ++connections_;
        std::thread([this, fd] { serve(fd); ::close(fd); --connections_; }).detach();
    }
    if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
}

void HttpServer::dispatch(const Request & request, Response & response) {
    if (request.method == "OPTIONS") {
        response.header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        response.header("Access-Control-Allow-Headers", "*");
        response.header("Access-Control-Max-Age", "86400");
        response.send(204, "", "");
        return;
    }
    if (before && !before(request, response)) return;
    bool path_known = false;
    for (const auto & route : routes_) {
        const bool prefix = !route.path.empty() && route.path.back() == '*';
        const bool match = prefix ? request.path.compare(0, route.path.size() - 1, route.path, 0, route.path.size() - 1) == 0
                                  : request.path == route.path;
        if (!match) continue;
        path_known = true;
        if (route.method != request.method && !(request.method == "HEAD" && route.method == "GET")) continue;
        route.handler(request, response);
        return;
    }
    response.send(path_known ? 405 : 404, "application/json; charset=utf-8", path_known
        ? "{\"error\":{\"code\":405,\"message\":\"Method not allowed\",\"type\":\"invalid_request_error\"}}"
        : "{\"error\":{\"code\":404,\"message\":\"File Not Found\",\"type\":\"not_found_error\"}}");
}

void HttpServer::serve(int fd) {
    Reader reader{fd, {}};
    for (;;) {
        Request request;
        std::string line;
        // Blank lines before a request line are tolerated (RFC 9112 2.2).
        do { if (!reader.line(line, 65536)) return; } while (line.empty());
        const auto first = line.find(' '), second = line.rfind(' ');
        if (first == std::string::npos || second == first) return;
        request.method = line.substr(0, first);
        const std::string version = line.substr(second + 1);
        std::string target = line.substr(first + 1, second - first - 1);
        const auto question = target.find('?');
        if (question != std::string::npos) { request.query = target.substr(question + 1); target.resize(question); }
        request.path = percent_decode(target);
        size_t header_bytes = 0;
        for (;;) {
            if (!reader.line(line, 65536) || (header_bytes += line.size()) > 1048576) return;
            if (line.empty()) break;
            const auto colon = line.find(':');
            if (colon == std::string::npos) continue;
            request.headers[lower(line.substr(0, colon))] = std::string(strip(std::string_view(line).substr(colon + 1)));
        }
        const bool http11 = version == "HTTP/1.1";
        const std::string connection = lower(request.header("connection"));
        const bool keep_alive = http11 ? connection.find("close") == std::string::npos : connection.find("keep-alive") != std::string::npos;
        Response response(fd, keep_alive, http11, request.method == "HEAD");
        if (lower(request.header("expect")) == "100-continue") {
            const char * interim = "HTTP/1.1 100 Continue\r\n\r\n";
            if (::send(fd, interim, std::strlen(interim), MSG_NOSIGNAL) < 0) return;
        }
        bool body_ok = true;
        if (lower(request.header("transfer-encoding")).find("chunked") != std::string::npos) {
            for (;;) {
                size_t size = 0;
                if (!reader.line(line, 4096)) return;
                const auto end = line.find(';');
                const std::string_view digits = strip(std::string_view(line).substr(0, end));
                if (digits.empty() || std::from_chars(digits.data(), digits.data() + digits.size(), size, 16).ptr != digits.data() + digits.size()) return;
                if (size > max_body || request.body.size() + size > max_body) { body_ok = false; break; }
                if (!size) { while (reader.line(line, 65536) && !line.empty()) {} break; }
                if (!reader.exact(request.body, size) || !reader.line(line, 16)) return;
            }
        } else if (const auto & length = request.header("content-length"); !length.empty()) {
            size_t size = 0;
            if (std::from_chars(length.data(), length.data() + length.size(), size).ptr != length.data() + length.size()) return;
            if (size > max_body) body_ok = false;
            else if (!reader.exact(request.body, size)) return;
        }
        const auto start = std::chrono::steady_clock::now();
        if (!body_ok) {
            response.send(413, "application/json; charset=utf-8", "{\"error\":{\"code\":413,\"message\":\"Request body too large\",\"type\":\"invalid_request_error\"}}");
            return;
        }
        try { dispatch(request, response); }
        catch (const std::exception & e) {
            // Handlers report their own errors; this is a last resort.
            response.send(500, "text/plain", std::string("Internal error: ") + e.what() + "\n");
        }
        if (!response.started()) response.send(500, "text/plain", "No response\n");
        response.end();
        if (after) after(request, response, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        if (!response.keep_alive()) return;
    }
}

} // namespace tidal::server
