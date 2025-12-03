#include <catch2/catch_test_macros.hpp>
#include <cnda/contiguous_nd.hpp>
#include <vector>
#include <memory>

TEST_CASE("non-owning view constructor basic functionality", "[view]") {
    // Create external buffer
    std::vector<int> external_buffer = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    
    // Create shared_ptr to manage the buffer
    auto owner = std::make_shared<std::vector<int>>(external_buffer);
    
    // Create non-owning view with shape [3, 4]
    cnda::ContiguousND<int> view({3, 4}, owner->data(), owner);
    
    REQUIRE(view.ndim() == 2);
    REQUIRE(view.size() == 12);
    REQUIRE(view.shape()[0] == 3);
    REQUIRE(view.shape()[1] == 4);
    REQUIRE(view.strides()[0] == 4);
    REQUIRE(view.strides()[1] == 1);
    REQUIRE(view.is_view() == true);
}

TEST_CASE("non-owning view reads correct values from external buffer", "[view]") {
    // Create and populate external buffer
    std::vector<double> external_buffer(12);
    for (std::size_t i = 0; i < 12; ++i) {
        external_buffer[i] = static_cast<double>(i * 10);
    }
    
    auto owner = std::make_shared<std::vector<double>>(external_buffer);
    
    // Create view with shape [3, 4]
    cnda::ContiguousND<double> view({3, 4}, owner->data(), owner);
    
    // Verify reading values through view
    REQUIRE(view(0, 0) == 0.0);
    REQUIRE(view(0, 1) == 10.0);
    REQUIRE(view(1, 2) == 60.0);
    REQUIRE(view(2, 3) == 110.0);
}

TEST_CASE("non-owning view can modify external buffer", "[view]") {
    // Create external buffer
    std::vector<int> external_buffer(12, 0);
    auto owner = std::make_shared<std::vector<int>>(external_buffer);
    
    // Create view
    cnda::ContiguousND<int> view({3, 4}, owner->data(), owner);
    
    // Modify through view
    view(0, 0) = 42;
    view(1, 2) = 99;
    view(2, 3) = 777;
    
    // Verify modifications in original buffer
    REQUIRE((*owner)[0] == 42);
    REQUIRE((*owner)[6] == 99);   // (1,2) with strides [4,1] => 1*4 + 2 = 6
    REQUIRE((*owner)[11] == 777); // (2,3) => 2*4 + 3 = 11
}

TEST_CASE("non-owning view shares data with external buffer", "[view]") {
    // Create external buffer
    std::vector<float> external_buffer = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto owner = std::make_shared<std::vector<float>>(external_buffer);
    
    // Create view with shape [2, 3]
    cnda::ContiguousND<float> view({2, 3}, owner->data(), owner);
    
    // Modify through view
    view(0, 1) = 999.0f;
    
    // Check that external buffer is modified
    REQUIRE((*owner)[1] == 999.0f);
    
    // Modify external buffer directly
    (*owner)[5] = 888.0f;
    
    // Check that view sees the change
    REQUIRE(view(1, 2) == 888.0f);
}

TEST_CASE("non-owning view data() points to external buffer", "[view]") {
    std::vector<int> external_buffer = {10, 20, 30, 40};
    auto owner = std::make_shared<std::vector<int>>(external_buffer);
    
    cnda::ContiguousND<int> view({2, 2}, owner->data(), owner);
    
    // data() should point to the same memory as external buffer
    REQUIRE(view.data() == owner->data());
    
    // Verify content
    REQUIRE(view.data()[0] == 10);
    REQUIRE(view.data()[3] == 40);
}

TEST_CASE("non-owning view with different shapes", "[view][shape]") {
    SECTION("1D view") {
        std::vector<int> buffer = {1, 2, 3, 4, 5};
        auto owner = std::make_shared<std::vector<int>>(buffer);
        
        cnda::ContiguousND<int> view({5}, owner->data(), owner);
        
        REQUIRE(view.ndim() == 1);
        REQUIRE(view.size() == 5);
        REQUIRE(view.strides()[0] == 1);
        REQUIRE(view.is_view() == true);
        
        REQUIRE(view(0) == 1);
        REQUIRE(view(4) == 5);
    }
    
    SECTION("3D view") {
        std::vector<int> buffer(24); // 2*3*4 = 24
        for (std::size_t i = 0; i < 24; ++i) {
            buffer[i] = static_cast<int>(i);
        }
        auto owner = std::make_shared<std::vector<int>>(buffer);
        
        cnda::ContiguousND<int> view({2, 3, 4}, owner->data(), owner);
        
        REQUIRE(view.ndim() == 3);
        REQUIRE(view.size() == 24);
        REQUIRE(view.strides()[0] == 12);
        REQUIRE(view.strides()[1] == 4);
        REQUIRE(view.strides()[2] == 1);
        REQUIRE(view.is_view() == true);
        
        // (1,1,2) => 1*12 + 1*4 + 2 = 18
        REQUIRE(view(1, 1, 2) == 18);
    }
    
    SECTION("4D view") {
        std::vector<double> buffer(120, 0.0); // 2*3*4*5 = 120
        auto owner = std::make_shared<std::vector<double>>(buffer);
        
        cnda::ContiguousND<double> view({2, 3, 4, 5}, owner->data(), owner);
        
        REQUIRE(view.ndim() == 4);
        REQUIRE(view.size() == 120);
        REQUIRE(view.is_view() == true);
        
        view(1, 2, 3, 4) = 3.14;
        REQUIRE((*owner)[119] == 3.14); // (1,2,3,4) => 1*60+2*20+3*5+4 = 119
    }
}

TEST_CASE("owning constructor is_view() returns false", "[view]") {
    // Regular owning constructor
    cnda::ContiguousND<int> owned({3, 4});
    
    REQUIRE(owned.is_view() == false);
    REQUIRE(owned.data() != nullptr);
}

TEST_CASE("non-owning view index() works correctly", "[view][index]") {
    std::vector<int> buffer(12);
    for (std::size_t i = 0; i < 12; ++i) {
        buffer[i] = static_cast<int>(i * 100);
    }
    auto owner = std::make_shared<std::vector<int>>(buffer);
    
    cnda::ContiguousND<int> view({3, 4}, owner->data(), owner);
    
    // Test index() computation
    REQUIRE(view.index({0, 0}) == 0);
    REQUIRE(view.index({1, 2}) == 6);
    REQUIRE(view.index({2, 3}) == 11);
    
    // Verify values using computed indices
    REQUIRE(view.data()[view.index({1, 2})] == 600);
}

TEST_CASE("non-owning view with const access", "[view][const]") {
    std::vector<int> buffer = {1, 2, 3, 4, 5, 6};
    auto owner = std::make_shared<std::vector<int>>(buffer);
    
    cnda::ContiguousND<int> view({2, 3}, owner->data(), owner);
    
    const auto& const_view = view;
    
    REQUIRE(const_view(0, 0) == 1);
    REQUIRE(const_view(1, 2) == 6);
    REQUIRE(const_view.data()[0] == 1);
    REQUIRE(const_view.is_view() == true);
}

TEST_CASE("multiple non-owning views can share same buffer", "[view]") {
    std::vector<int> buffer = {10, 20, 30, 40, 50, 60};
    auto owner = std::make_shared<std::vector<int>>(buffer);
    
    // Create two views of the same buffer with different shapes
    cnda::ContiguousND<int> view1({2, 3}, owner->data(), owner);
    cnda::ContiguousND<int> view2({6}, owner->data(), owner);
    
    REQUIRE(view1.is_view() == true);
    REQUIRE(view2.is_view() == true);
    
    // Both views should see the same data
    REQUIRE(view1(0, 0) == 10);
    REQUIRE(view2(0) == 10);
    
    // Modify through view1
    view1(1, 2) = 999;
    
    // view2 should see the change (element 5 in 1D = element (1,2) in 2D shape)
    REQUIRE(view2(5) == 999);
    REQUIRE((*owner)[5] == 999);
}

TEST_CASE("non-owning view with raw pointer and custom deleter", "[view]") {
    // Allocate raw buffer
    int* raw_buffer = new int[8];
    for (int i = 0; i < 8; ++i) {
        raw_buffer[i] = i * 10;
    }
    
    // Create shared_ptr with custom deleter
    auto owner = std::shared_ptr<void>(raw_buffer, [](void* p) {
        delete[] static_cast<int*>(p);
    });
    
    cnda::ContiguousND<int> view({2, 4}, raw_buffer, owner);
    
    REQUIRE(view.is_view() == true);
    REQUIRE(view.size() == 8);
    REQUIRE(view(0, 0) == 0);
    REQUIRE(view(1, 3) == 70);
    
    // Modify through view
    view(0, 2) = 555;
    REQUIRE(raw_buffer[2] == 555);
}

TEST_CASE("non-owning view with zero-sized dimension", "[view][edge]") {
    std::vector<int> buffer; // empty buffer
    auto owner = std::make_shared<std::vector<int>>(buffer);
    
    cnda::ContiguousND<int> view({0, 5}, owner->data(), owner);
    
    REQUIRE(view.is_view() == true);
    REQUIRE(view.size() == 0);
    REQUIRE(view.ndim() == 2);
    REQUIRE(view.shape()[0] == 0);
    REQUIRE(view.shape()[1] == 5);
}

#ifdef CNDA_BOUNDS_CHECK
TEST_CASE("non-owning view bounds checking", "[view][bounds]") {
    std::vector<int> buffer(12, 0);
    auto owner = std::make_shared<std::vector<int>>(buffer);
    
    cnda::ContiguousND<int> view({3, 4}, owner->data(), owner);
    
    SECTION("Out of bounds access throws") {
        REQUIRE_THROWS_AS(view(3, 0), std::out_of_range);
        REQUIRE_THROWS_AS(view(0, 4), std::out_of_range);
        REQUIRE_THROWS_AS(view(3, 4), std::out_of_range);
    }
    
    SECTION("Rank mismatch throws") {
        REQUIRE_THROWS_AS(view.index({0}), std::out_of_range);
        REQUIRE_THROWS_AS(view.index({0, 0, 0}), std::out_of_range);
        REQUIRE_THROWS_AS(view(0), std::out_of_range);
        REQUIRE_THROWS_AS(view(0, 0, 0), std::out_of_range);
    }
}
#endif

TEST_CASE("non-owning view lifetime management", "[view][lifetime]") {
    // Create a view and let the owner go out of scope in a controlled way
    std::vector<int> buffer = {1, 2, 3, 4};
    auto owner = std::make_shared<std::vector<int>>(buffer);
    
    cnda::ContiguousND<int> view({2, 2}, owner->data(), owner);
    
    REQUIRE(view(0, 0) == 1);
    REQUIRE(view(1, 1) == 4);
    
    // owner use_count should be 2 (owner variable + view's m_external_owner)
    REQUIRE(owner.use_count() == 2);
}
