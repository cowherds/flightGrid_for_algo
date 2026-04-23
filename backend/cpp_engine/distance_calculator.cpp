#include <cmath>
#include <cstddef>

#ifdef DISTANCE_USE_OPENMP
#include <omp.h>
#endif

namespace {

constexpr double kEpsilon = 1e-9;

struct Point2D {
    double x;
    double y;
};

double cross(const Point2D& a, const Point2D& b, const Point2D& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool on_segment(const Point2D& a, const Point2D& b, const Point2D& p) {
    if (std::fabs(cross(a, b, p)) > kEpsilon) {
        return false;
    }
    const double min_x = (a.x < b.x) ? a.x : b.x;
    const double max_x = (a.x > b.x) ? a.x : b.x;
    const double min_y = (a.y < b.y) ? a.y : b.y;
    const double max_y = (a.y > b.y) ? a.y : b.y;
    return p.x >= min_x - kEpsilon && p.x <= max_x + kEpsilon &&
           p.y >= min_y - kEpsilon && p.y <= max_y + kEpsilon;
}

int orientation(const Point2D& a, const Point2D& b, const Point2D& c) {
    const double value = cross(a, b, c);
    if (std::fabs(value) <= kEpsilon) {
        return 0;
    }
    return (value > 0.0) ? 1 : 2;
}

bool segments_intersect(const Point2D& p1, const Point2D& p2, const Point2D& q1, const Point2D& q2) {
    const int o1 = orientation(p1, p2, q1);
    const int o2 = orientation(p1, p2, q2);
    const int o3 = orientation(q1, q2, p1);
    const int o4 = orientation(q1, q2, p2);

    if (o1 != o2 && o3 != o4) {
        return true;
    }

    if (o1 == 0 && on_segment(p1, p2, q1)) {
        return true;
    }
    if (o2 == 0 && on_segment(p1, p2, q2)) {
        return true;
    }
    if (o3 == 0 && on_segment(q1, q2, p1)) {
        return true;
    }
    if (o4 == 0 && on_segment(q1, q2, p2)) {
        return true;
    }
    return false;
}

bool point_in_polygon(
    const Point2D& point,
    const double* polygons_xy,
    int start_vertex,
    int vertex_count
) {
    if (vertex_count < 3) {
        return false;
    }

    bool inside = false;
    int prev = start_vertex + vertex_count - 1;
    for (int i = 0; i < vertex_count; ++i) {
        const int curr = start_vertex + i;
        const Point2D a{
            polygons_xy[2 * curr],
            polygons_xy[2 * curr + 1],
        };
        const Point2D b{
            polygons_xy[2 * prev],
            polygons_xy[2 * prev + 1],
        };

        if (on_segment(a, b, point)) {
            return true;
        }

        const bool intersects = ((a.y > point.y) != (b.y > point.y)) &&
                                (point.x < (b.x - a.x) * (point.y - a.y) /
                                               ((b.y - a.y) == 0.0 ? kEpsilon : (b.y - a.y)) +
                                           a.x);
        if (intersects) {
            inside = !inside;
        }
        prev = curr;
    }
    return inside;
}

bool segment_intersects_polygon(
    const Point2D& start,
    const Point2D& end,
    const double* polygons_xy,
    int start_vertex,
    int vertex_count
) {
    if (vertex_count < 3) {
        return false;
    }

    if (point_in_polygon(start, polygons_xy, start_vertex, vertex_count) ||
        point_in_polygon(end, polygons_xy, start_vertex, vertex_count)) {
        return true;
    }

    int prev = start_vertex + vertex_count - 1;
    for (int i = 0; i < vertex_count; ++i) {
        const int curr = start_vertex + i;
        const Point2D edge_a{
            polygons_xy[2 * prev],
            polygons_xy[2 * prev + 1],
        };
        const Point2D edge_b{
            polygons_xy[2 * curr],
            polygons_xy[2 * curr + 1],
        };

        if (segments_intersect(start, end, edge_a, edge_b)) {
            return true;
        }
        prev = curr;
    }
    return false;
}

}  // namespace

extern "C" void calculate_distance_matrix(
    const double* points_xy,
    int num_points,
    const double* polygons_xy,
    const int* polygon_vertex_counts,
    int num_polygons,
    double blocked_distance,
    double* out_matrix
) {
    if (out_matrix == nullptr || points_xy == nullptr || num_points <= 0) {
        return;
    }

#ifdef DISTANCE_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < num_points; ++i) {
        const Point2D pi{
            points_xy[2 * i],
            points_xy[2 * i + 1],
        };

        for (int j = 0; j < num_points; ++j) {
            const Point2D pj{
                points_xy[2 * j],
                points_xy[2 * j + 1],
            };

            const std::size_t out_index = static_cast<std::size_t>(i) * static_cast<std::size_t>(num_points) +
                                          static_cast<std::size_t>(j);

            if (i == j) {
                out_matrix[out_index] = 0.0;
                continue;
            }

            bool blocked = false;
            if (polygons_xy != nullptr && polygon_vertex_counts != nullptr && num_polygons > 0) {
                int vertex_cursor = 0;
                for (int p = 0; p < num_polygons; ++p) {
                    const int vertex_count = polygon_vertex_counts[p];
                    if (vertex_count >= 3) {
                        if (segment_intersects_polygon(pi, pj, polygons_xy, vertex_cursor, vertex_count)) {
                            blocked = true;
                            break;
                        }
                    }
                    vertex_cursor += vertex_count;
                }
            }

            if (blocked) {
                out_matrix[out_index] = blocked_distance;
                continue;
            }

            const double dx = pi.x - pj.x;
            const double dy = pi.y - pj.y;
            out_matrix[out_index] = std::sqrt(dx * dx + dy * dy);
        }
    }
}

extern "C" void calculate_segment_blocked_flags(
    const double* segments_xy,
    int num_segments,
    const double* polygons_xy,
    const int* polygon_vertex_counts,
    int num_polygons,
    unsigned char* out_blocked
) {
    if (out_blocked == nullptr || segments_xy == nullptr || num_segments <= 0) {
        return;
    }

#ifdef DISTANCE_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int s = 0; s < num_segments; ++s) {
        const Point2D start{
            segments_xy[4 * s],
            segments_xy[4 * s + 1],
        };
        const Point2D end{
            segments_xy[4 * s + 2],
            segments_xy[4 * s + 3],
        };

        bool blocked = false;
        if (polygons_xy != nullptr && polygon_vertex_counts != nullptr && num_polygons > 0) {
            int vertex_cursor = 0;
            for (int p = 0; p < num_polygons; ++p) {
                const int vertex_count = polygon_vertex_counts[p];
                if (vertex_count >= 3) {
                    if (segment_intersects_polygon(start, end, polygons_xy, vertex_cursor, vertex_count)) {
                        blocked = true;
                        break;
                    }
                }
                vertex_cursor += vertex_count;
            }
        }

        out_blocked[s] = blocked ? static_cast<unsigned char>(1) : static_cast<unsigned char>(0);
    }
}
