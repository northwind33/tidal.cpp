#pragma once
// Small blocking HTTP/1.1 server: one thread per connection, keep-alive,
// fixed-length and chunked request bodies, and chunked streaming responses.
#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace tidal::server {

struct Request {
    std::string method, path, query, body;
    std::map<std::string, std::string> headers; // Lower-case names.
    const std::string & header(const std::string & name) const;
    // Value of a query-string parameter, percent-decoded; empty if absent.
    std::string parameter(std::string_view name) const;
};

class Response {
public:
    // `head` answers a HEAD request: headers only.
    Response(int fd, bool keep_alive, bool chunked, bool head = false)
        : fd_(fd), keep_alive_(keep_alive), chunked_(chunked), head_(head) {}
    // Extra header for the next send() or begin().
    void header(std::string_view name, std::string_view value);
    // A complete response. Ignored once a response has started.
    void send(int status, std::string_view content_type, std::string_view body);
    // Starts a streamed response; write() then sends pieces as they are ready.
    bool begin(int status, std::string_view content_type);
    bool write(std::string_view data);
    void end();
    // False once the peer has closed the connection or a write failed.
    bool alive();
    bool started() const { return started_; }
    bool streaming() const { return streaming_; }
    int status() const { return status_; }
    bool keep_alive() const { return keep_alive_ && !failed_; }
private:
    int fd_, status_ = 0;
    bool keep_alive_, chunked_, head_, started_ = false, streaming_ = false, failed_ = false;
    std::string extra_;
    bool raw(std::string_view data);
    std::string head(int status, std::string_view content_type) const;
};

using Handler = std::function<void(const Request &, Response &)>;

class HttpServer {
public:
    // A path ending in '*' matches every path with that prefix.
    void route(const std::string & method, const std::string & path, Handler handler);
    // Runs before the route; returning false means it already responded.
    std::function<bool(const Request &, Response &)> before;
    // Called after each request with the elapsed milliseconds.
    std::function<void(const Request &, const Response &, double)> after;
    size_t max_body = size_t(1) << 30;
    // Binds and listens; throws on failure. Port 0 picks a free port.
    void listen(const std::string & host, int port);
    int port() const { return port_; }
    // Accepts connections until stop().
    void run();
    // Safe to call from a signal handler.
    void stop();
private:
    struct Route { std::string method, path; Handler handler; };
    std::vector<Route> routes_;
    int listen_fd_ = -1, port_ = 0;
    std::atomic<bool> stopping_{false};
    std::atomic<int> connections_{0};
    void serve(int fd);
    void dispatch(const Request & request, Response & response);
};

} // namespace tidal::server
